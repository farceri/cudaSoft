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
  bool readState = true, saveFinal = true, linSave = true;
  bool lj = true, wca = false, alltoall = false, testNVT = false;
  bool fixedbc = false, roundbc = true, reflect = false, reflectnoise = false;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10);
  long numParticles = atol(argv[5]), nDim = 2, linFreq = int(checkPointFreq / 10);
  long saveEnergyFreq = int(checkPointFreq / 10), updateCount = 0, totUpdate = 0;
  double ec = 1, ew = 1, cutDistance, sigma, timeStep = atof(argv[2]), Tinject = atof(argv[3]);
  double LJcut = 4, cutoff = 0.5, iod = sqrt(10), damping, timeUnit;
  std::string energyFile, outDir, inDir = argv[1], currentDir;
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setEnergyCostant(ec);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
  } else {
    sp.setBoxType(simControlStruct::boxEnum::harmonic);
    cout << "Setting Harmonic potential" << endl;
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
    sp.setBoxEnergyScale(ew);
    cout << "Setting fixed rectangular boundary conditins" << endl;
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundBox);
    sp.setBoxEnergyScale(ew);
    cout << "Setting fixed circular boundary conditins" << endl;
  } else {
    cout << "Setting periodic boundary conditins" << endl;
  }
  if(fixedbc == true || roundbc == true) {
    if(reflect == true) {
      sp.setBoxType(simControlStruct::boxEnum::reflect);
    cout << "Setting reflective walls" << endl;
    } else if(reflectnoise == true) {
      sp.setBoxType(simControlStruct::boxEnum::reflectnoise);
    cout << "Setting reflective walls with noise" << endl;
    } else {
      cout << "Setting repulsive walls" << endl;
    }
  }
  ioSPFile ioSP(&sp);
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // initialization
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  if(testNVT == true) {
    outDir = inDir + "testNVT/";
    std::experimental::filesystem::create_directory(outDir);
    sigma = 2 * sp.getMeanParticleSigma();
    damping = sqrt(iod) / sigma;
    timeUnit = sigma / sqrt(ec);
    timeStep = sp.setTimeStep(timeStep * timeUnit);
    cout << "NVT: time step: " << timeStep << " damping: " << damping << " sigma: " << sigma << endl;
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  } else {
    outDir = inDir + "testNVE";
    if(reflect == true) {
      outDir = outDir + "-reflect/";
    } else if(reflectnoise == true) {
      outDir = outDir + "-reflectnoise/";
    } else {
      outDir = outDir + "/";
    }
    std::experimental::filesystem::create_directory(outDir);
    sigma = 2 * sp.getMeanParticleSigma();
    timeUnit = sigma / sqrt(ec);//mass is 1 - sqrt(m sigma^2 / epsilon)
    timeStep = sp.setTimeStep(timeStep * timeUnit);
    cout << "NVE: time step: " << timeStep << " sigma: " << sigma << endl;
    sp.initSoftParticleNVE(Tinject, readState);
  }
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  sp.resetLastPositions();
  while(step != maxStep) {
    //cout << "step: " << step << endl;
    if(testNVT == true) {
      sp.softParticleLangevinLoop();
    } else {
      sp.softParticleNVELoop();
    }
    if(step % saveEnergyFreq == 0 && step > 0) {
      ioSP.saveSimpleEnergy(step, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Test: current step: " << step;
        cout << " E: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no neighbor updates in this simulation block" << endl;
        }
        sp.resetUpdateCount();
        if(saveFinal == true) {
          ioSP.saveParticlePacking(outDir);
        }
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(step) + "/";
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
  // save last configuration
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();
  return 0;
}
