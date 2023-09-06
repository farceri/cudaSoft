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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave, linSave;
  long numParticles = atol(argv[7]), nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), updateFreq = 10;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = 1e03, firstDecade = 0;
  double cutDistance = 1, waveQ, sigma, damping, inertiaOverDamping = atof(argv[6]);
  double ec = 2400, timeUnit, timeStep = atof(argv[2]), Tinject = atof(argv[3]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  //dirSample = whichDynamics + "T" + argv[3] + "/iod" + argv[6] + "/";
  // initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      //logSave = true;
      linSave = true;
      //outDir = outDir + "dynamics/";
      outDir = outDir + "dynamics-k10/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        initialStep = atof(argv[5]);
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
  ioDPM.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioDPM.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  dpm.setEnergyCosts(0, 0, 0, ec);
  sigma = dpm.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  //timeUnit = sigma * sigma * damping;
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  //timeStep = dpm.setTimeStep(timeStep);
  cout << "Inertia over damping: " << inertiaOverDamping << ", damping: " << damping << endl;
  cout << "Tinject: " << Tinject << " time step: " << timeStep << " sigma: " << sigma << endl;
  ioDPM.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  // initialize simulation
  dpm.calcParticleNeighborList(cutDistance);
  dpm.calcParticleForceEnergy();
  dpm.initSoftParticleLangevin(Tinject, damping, readState);
  // run integrator
  waveQ = dpm.getSoftWaveNumber();
  while(step != maxStep) {
    dpm.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0 && step > 0) {
      ioDPM.saveParticleEnergy(step, timeStep, waveQ);
      if(step % checkPointFreq == 0) {
        cout << "Brownian: current step: " << step;
        cout << " U/N: " << dpm.getParticleEnergy() / numParticles;
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ) << endl;
        ioDPM.saveParticleConfiguration(outDir);
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
        ioDPM.saveParticleState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveParticleState(currentDir);
      }
    }
    if(step % updateFreq == 0) {
      dpm.calcParticleNeighborList(cutDistance);
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioDPM.saveParticleConfiguration(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
