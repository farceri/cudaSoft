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
#include <tuple>
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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true, ljwca = false, ljmp = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave = false, linSave = true, alltoall = false, fixedbc = false;
  long numParticles = atol(argv[7]), nDim = atol(argv[8]), maxStep = atof(argv[5]), num1 = atol(argv[9]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[6]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, LJcut = 4, cutoff = 0.5, cutDistance, waveQ, timeStep = atof(argv[2]), timeUnit, sigma, mass = 10;
  double ea = 1, eb = 1, eab = 0.1, Tinject = atof(argv[3]), Tinject2 = atof(argv[4]), damping = 1, damping2 = 1;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "2T-2lj/";
  std::tuple<double, double, double> Temps;
  if(ljwca == true) {
    whichDynamics = "2T-ljwca/";
  } else if(ljmp == true) {
    whichDynamics = "2T-ljmp/";
  }
  dirSample = whichDynamics + "T1-" + argv[3] + "-T2-" + argv[4] + "/";
  if(nDim == 3) {
    LJcut = 2.5;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
  }
  if(ljwca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::LJWCA);
    sp.setEnergyCostant(ec);
    sp.setLJWCAparams(LJcut, num1);
  } else if(ljmp == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::LJMinusPlus);
    sp.setEnergyCostant(ec);
    sp.setLJMinusPlusParams(LJcut, num1);
  } else {
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  ioSPFile ioSP(&sp);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      if(logSave == true) {
        outDir = outDir + "dynamics-log/";
      } else {
        outDir = outDir + "dynamics/";
      }
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
    ioSP.readDoubleNoseHooverParams(inDir, mass, damping, damping2);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << endl;
  cout << "T1: " << Tinject << " T2: " << Tinject2 << " time step: " << timeStep << endl;
  // initialize simulation
  sp.initSoftParticleDoubleNoseHoover(Tinject, Tinject2, mass, damping, damping2, readState);
  //sp.initSoftParticleNoseHoover(Tinject, mass, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  sp.setInitialPositions();
  waveQ = sp.getSoftWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  while(step != maxStep) {
    sp.softParticleDoubleNoseHooverLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticleDoubleNoseHooverEnergy(step+initialStep, timeStep, numParticles, num1);
      if(step % checkPointFreq == 0) {
        cout << "Double NH: current step: " << step;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        Temps = sp.getParticleT1T2();
        cout << " T1: " << get<0>(Temps) << " T2: " << get<1>(Temps) << " T: " << get<2>(Temps);
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
          ioSP.saveParticleNeighbors(outDir);
          ioSP.saveDoubleNoseHooverParams(outDir);
          if(nDim == 3) {
            ioSP.saveDumpPacking(outDir, numParticles, nDim, step);
          }
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
        ioSP.saveDoubleNoseHooverParams(outDir);
        //ioSP.saveParticleNeighbors(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
        ioSP.saveDoubleNoseHooverParams(outDir);
        ioSP.saveParticleNeighbors(currentDir);
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
    ioSP.saveParticleNeighbors(outDir);
    ioSP.saveDoubleNoseHooverParams(outDir);
    if(nDim == 3) {
      ioSP.saveDumpPacking(outDir, numParticles, nDim, step);
    }
  }
  ioSP.closeEnergyFile();

  return 0;
}
