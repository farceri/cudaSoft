//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Include C++ header files

#include "include/DPM2D.h"
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
  bool readState = true, readAndSaveSameDir = false, testNVE = true, testNVT = false, update = false;
  long numParticles = atol(argv[5]), nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10);
  long saveEnergyFreq = int(checkPointFreq / 10), updateCount = 0, updateFreq = 100, totUpdate = 0;
  double ec = 240, cutDistance = 1, sigma, timeStep = atof(argv[2]), Tinject = atof(argv[3]);
  double Dr = 2e-04, driving = 8.5e-02, iod = 10, damping, timeUnit, forceUnit, cutoff, maxDelta;
  std::string energyFile, outDir, inDir = argv[1], currentDir;
  // initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  ioDPM.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioDPM.readParticleState(inDir, numParticles, nDim);
  }
  // initialization
  dpm.setEnergyCosts(0, 0, 0, ec);
  sigma = dpm.getMeanParticleSigma();
  // initialize simulation
  dpm.calcParticleNeighborList(cutDistance);
  dpm.calcParticleForceEnergy();
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
    timeStep = dpm.setTimeStep(timeStep * timeUnit);
    cout << "NVE: time step: " << timeStep << " sigma: " << sigma << endl;
    dpm.initSoftParticleNVE(Tinject, readState);
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
    timeStep = dpm.setTimeStep(timeStep * timeUnit);
    cout << "NVT: time step: " << timeStep << " damping: " << damping << " sigma: " << sigma << endl;
    dpm.initSoftParticleLangevin(Tinject, damping, readState);
  }
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  cutoff = cutDistance * dpm.getMinParticleSigma();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  dpm.resetPreviousPositions();
  while(step != maxStep) {
    //cout << "step: " << step << endl;
    if(testNVE == true) {
      dpm.softParticleNVELoop();
    }
    else if(testNVT == true) {
      dpm.softParticleLangevinLoop();
    }
    if(step % saveEnergyFreq == 0 && step > 0) {
      ioDPM.saveParticleSimpleEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "Test: current step: " << step;
        cout << " E: " << (dpm.getParticleEnergy() + dpm.getParticleKineticEnergy()) / numParticles;
        cout << " T: " << dpm.getParticleTemperature() << endl;
        //ioDPM.saveParticleConfiguration(outDir);
        if(step != 0 && updateCount > 0) {
          cout << "Number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << "No updates in this simulation block" << endl;
        }
        updateCount = 0;
      }
    }
    if(update == true) {
      maxDelta = dpm.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        totUpdate += 1;
        dpm.calcParticleNeighborList(cutDistance);
        dpm.resetPreviousPositions();
        updateCount += 1;
      }
    } else {
      if(step % updateFreq == 0) {
        maxDelta = dpm.getParticleMaxDisplacement();
        dpm.calcParticleNeighborList(cutDistance);
        dpm.resetPreviousPositions();
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
  ioDPM.saveParticleConfiguration(outDir);

  ioDPM.closeEnergyFile();
  return 0;
}
