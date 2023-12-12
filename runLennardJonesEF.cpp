//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Include C++ header files

#include "include/SP2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
#include "include/defs.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <functional>
#include <utility>
#include <thrust/host_vector.h>
#include <experimental/filesystem>

using namespace std;

int main(int argc, char **argv) {
  // variables
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave, linSave, perturb = false;
  long numParticles = atol(argv[7]), nDim = 2, maxIndex = atol(argv[9]);
  long maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), saveEnergyFreq = int(checkPointFreq / 10);
  long linFreq = 1e03, initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;//, updateFreq = 10;
  double ec = 1, Tinject = atof(argv[3]), cutoff, LJcut = 5.5, sigma, timeUnit, timeStep = atof(argv[2]);
  double cutDistance = LJcut-0.5, maxDelta, waveQ, damping, inertiaOverDamping = atof(argv[6]), externalForce = atof(argv[8]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin-lj/";
  dirSample = whichDynamics + "T" + argv[3] + "-EF" + argv[9] + "/";
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
  ioSPFile ioSP(&sp);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      //logSave = true;
      //outDir = outDir + "dynamics-log/";
      linSave = true;
      if(perturb == true) {
        outDir = outDir + "perturb-f" + argv[8] + "/";
      } else {
        outDir = outDir + "perturb-f" + argv[8] + "/dynamics/";
        inDir = inDir + "perturb-f" + argv[8] + "/";
      }
      if(std::experimental::filesystem::exists(outDir) == true) {
        if(initialStep != 0) {
        inDir = outDir;
        }
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setEnergyCostant(ec);
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  sigma = sp.getMeanParticleSigma();
  externalForce *= inertiaOverDamping / sigma;
  sp.setLJcutoff(LJcut);
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << endl;
  cout << "Thermal energy scale: " << Tinject << endl;
  cout << "External force: " << externalForce << " applied to " << maxIndex << " particles" << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergyLJ();
  if(perturb == true) {
    sp.initSoftParticleLangevinPerturb(Tinject, damping, externalForce, maxIndex, readState);
  } else {
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  }
  // run integrator
  waveQ = sp.getSoftWaveNumber();
  sp.setInitialPositions();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  while(step != maxStep) {
    if(perturb == true) {
      sp.softParticleLangevinPerturbLoop();
    } else {
      sp.softParticleLangevinLoop();
    }
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticleSimpleEnergy(step+initialStep, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "LJ-EF: current step: " << step + initialStep;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        updateCount = 0;
        if(saveFinal == true) {
          ioSP.saveParticlePacking(outDir);
        }
      }
    }
    if(logSave == true) {
      if(step > (multiple * checkPointFreq)) {
        saveFreq = 1;
        multiple += 1;
      }
      if((step - (multiple-1) * checkPointFreq) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * checkPointFreq) % saveFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
      }
    }
    maxDelta = sp.getParticleMaxDisplacement();
    if(3*maxDelta > cutoff) {
      sp.calcParticleNeighborList(cutDistance);
      sp.resetLastPositions();
      updateCount += 1;
    }
    //if(step % updateFreq == 0) {
    //  sp.calcParticleNeighborList(cutDistance);
    //  updateCount += 1;
    //}
    step += 1;
  }
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
