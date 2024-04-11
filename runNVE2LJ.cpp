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
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave = false, linSave = false, scaleVel = false, doubleScaleVel = false;
  long numParticles = atol(argv[6]), nDim = 2, maxStep = atof(argv[4]), num1 = atol(argv[7]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double LJcut = 4, cutoff = 1, cutDistance, waveQ, timeStep = atof(argv[2]);
  double ea = 1, eb = 1, eab = 0.25, Tinject = atof(argv[3]), Tinject2 = atof(argv[8]), sigma, timeUnit;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "nve/";
  std::tuple<double, double> Temps;
  dirSample = whichDynamics + "/";//+ "T" + argv[3] + 
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
  sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
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
        //outDir = outDir + "T1-" + argv[3] + "-T2-" + argv[8] + "/";
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
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << endl;
  if(doubleScaleVel == true) {
    cout << "T1: " << Tinject << " T2: " << Tinject2 << " time step: " << timeStep << endl;
  } else if(readState == false) {
    cout << "Tinject: " << Tinject << " time step: " << timeStep << endl;
  } else {
    cout << "Reading state - time step: " << timeStep << endl;
  }
  // initialize simulation
  if(scaleVel == true) {
    sp.initSoftParticleNVERescale(Tinject);
  } else if(doubleScaleVel == true) {
    sp.initSoftParticleNVEDoubleRescale(Tinject, Tinject2);
  } else {
    sp.initSoftParticleNVE(Tinject, readState);
  }
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
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
  ioSP.saveParticlePacking(outDir);
  ioSP.saveParticleNeighbors(outDir);
  while(step != maxStep) {
    if(scaleVel == true) {
      sp.softParticleNVERescaleLoop();
    } else if(doubleScaleVel == true) {
      sp.softParticleNVEDoubleRescaleLoop();
    } else {
      sp.softParticleNVELoop();
    }
    if(step % saveEnergyFreq == 0) {
      if(doubleScaleVel == true) {
        ioSP.saveParticleDoubleEnergy(step+initialStep, timeStep, numParticles);
      } else {
        ioSP.saveParticleSimpleEnergy(step+initialStep, timeStep, numParticles);
      }
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        if(doubleScaleVel == true) {
          Temps = sp.getParticleT1T2();
          cout << " T1: " << get<0>(Temps) << " T2: " << get<1>(Temps);
        } else {
          cout << " T: " << sp.getParticleTemperature();
        }
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
        //ioSP.saveParticleNeighbors(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
        //ioSP.saveParticleNeighbors(currentDir);
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
  }
  ioSP.closeEnergyFile();

  return 0;
}
