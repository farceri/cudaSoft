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
  bool readState = true, readAndSaveSameDir = false, testNVE = true, testNVT = false, update = true;
  long numParticles = atol(argv[5]), nDim = 2;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10);
  long saveEnergyFreq = int(checkPointFreq / 10), updateCount = 0, updateFreq = 100, totUpdate = 0;
  double ec = 240, cutDistance = 1, sigma, timeStep = atof(argv[2]), Tinject = atof(argv[3]);
  double Dr = 2e-04, driving = 8.5e-02, iod = 10, damping, timeUnit, forceUnit, cutoff, maxDelta;
  std::string energyFile, outDir, inDir = argv[1], currentDir;
  // initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // initialization
  sp.setEnergyCostant(ec);
  sigma = sp.getMeanParticleSigma();
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  outDir = inDir;
  if(testNVE == true) {
    if(readAndSaveSameDir == false) {
      if(update == true){
        outDir = inDir + "/testNVE-update/";
      } else {
        outDir = inDir + "/testNVE/";
      }
    }
    std::experimental::filesystem::create_directory(outDir);
    timeUnit = sigma;
    timeStep = sp.setTimeStep(timeStep * timeUnit);
    cout << "NVE: time step: " << timeStep << " sigma: " << sigma << endl;
    sp.initSoftParticleNVE(Tinject, readState);
  }
  else if(testNVT == true) {
    if(readAndSaveSameDir == false) {
      if(update == true){
        outDir = inDir + "/testNVT-update/";
      } else {
        outDir = inDir + "/testNVT/";
      }
    }
    std::experimental::filesystem::create_directory(outDir);
    damping = sqrt(iod) / sigma;
    timeUnit = 1 / damping;
    timeStep = sp.setTimeStep(timeStep * timeUnit);
    cout << "NVT: time step: " << timeStep << " damping: " << damping << " sigma: " << sigma << endl;
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  }
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  cutoff = cutDistance * sp.getMinParticleSigma();
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
    if(testNVE == true) {
      sp.softParticleNVELoop();
    }
    else if(testNVT == true) {
      sp.softParticleLangevinLoop();
    }
    if(step % saveEnergyFreq == 0 && step > 0) {
      ioSP.saveParticleSimpleEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "Test: current step: " << step;
        cout << " E: " << (sp.getParticleEnergy() + sp.getParticleKineticEnergy()) / numParticles;
        cout << " T: " << sp.getParticleTemperature() << endl;
        if(step != 0 && updateCount > 0) {
          cout << "Number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << "No updates in this simulation block" << endl;
        }
        updateCount = 0;
      }
    }
    if(update == true) {
      maxDelta = sp.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        totUpdate += 1;
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
        updateCount += 1;
      }
    } else {
      if(step % updateFreq == 0) {
        maxDelta = sp.getParticleMaxDisplacement();
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
        updateCount += 1;
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
  ioSP.saveParticlePacking(outDir);

  ioSP.closeEnergyFile();
  return 0;
}
