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
  bool readState = true, saveFinal = true, logSave, linSave;
  long numParticles = atol(argv[7]), nDim = 2;
  long maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, n = 12, m = 6, LJcut = 5.5, cutDistance = LJcut+0.5, cutoff, waveQ, timeStep = atof(argv[2]);
  double Tinject = atof(argv[3]), damping, inertiaOverDamping = atof(argv[6]), sigma, forceUnit, timeUnit, range = 3;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin-lj/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
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
      outDir = outDir + "dynamics/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        //if(initialStep != 0) {
        inDir = outDir;
        //}
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
      //outDir = inDir + "../../../" + dirSample;
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
  sp.setLJcutoff(LJcut);
  //sp.setMieParams(LJcut, n, m);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  //timeUnit = 1 / damping;
  //forceUnit = inertiaOverDamping / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) * forceUnit << endl;
  damping /= timeUnit;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutoff = 2 * (1 + cutDistance) * sp.getMinParticleSigma();
  sp.setDisplacementCutoff(cutoff, cutDistance);
  sp.resetUpdateCount();
  sp.setInitialPositions();
  waveQ = sp.getSoftWaveNumber();
  range *= LJcut * sigma;
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  while(step != maxStep) {
    sp.softParticleLangevinLoop();
    if(step % linFreq == 0) {
      //ioSP.saveParticleSimpleEnergy(step+initialStep, timeStep, numParticles);
      ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
      if(step % checkPointFreq == 0) {
        cout << "Langevin LJ: current step: " << step + initialStep;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        sp.resetUpdateCount();
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
