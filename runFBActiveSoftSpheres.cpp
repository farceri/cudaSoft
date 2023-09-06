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
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, logSave = false, linSave = true, saveFinal = true;
  long numParticles = 8192, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), updateFreq = 10;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 10;
  long linFreq = 1e03, firstDecade = 0;
  double cutDistance = 1, waveQ, sigma, damping, forceUnit, timeUnit, timeStep = atof(argv[2]);
  double ec = 240, Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]), inertiaOverDamping = atof(argv[8]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-langevin/";
  //dirSample = whichDynamics + "T" + argv[3] + "/";
  dirSample = whichDynamics + "Dr" + argv[4] + "-f0" + argv[5] + "/";
  // initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics-fb/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        initialStep = atof(argv[7]);
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
    ioDPM.readParticleActiveState(inDir, numParticles, nDim);
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
  forceUnit = inertiaOverDamping / sigma;
  //forceUnit = 1 / (inertiaOverDamping * sigma);
  cout << "Inertia over damping: " << inertiaOverDamping << " damping: " << damping << " sigma: " << sigma << endl;
  cout << "Tinject: " << Tinject << " time step: " << timeStep << " taup: " << timeUnit/Dr << endl;
  cout << "Peclet number: " << driving * forceUnit * timeUnit / (damping * Dr * sigma);
  cout << " f0: " << driving*forceUnit << ", " << driving << " Dr: " << Dr/timeUnit << ", " << Dr << endl;
  driving = driving*forceUnit;
  Dr = Dr/timeUnit;
  ioDPM.saveParticleDynamicalParams(outDir, sigma, damping, Dr, driving);
  // initialize simulation
  dpm.calcParticleWallNeighborList(cutDistance);
  dpm.calcParticleWallForceEnergy();
  dpm.initSoftParticleActiveLangevinFixedBoundary(Tinject, Dr, driving, damping, readState);
  //dpm.initSoftParticleActiveLangevinFixedSides(Tinject, Dr, driving, damping, readState);
  // run integrator
  waveQ = dpm.getSoftWaveNumber();
  while(step != maxStep) {
    dpm.softParticleActiveLangevinFixedBoundaryLoop();
    //dpm.softParticleActiveLangevinFixedSidesLoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveParticleActiveEnergy(step, timeStep, waveQ, driving);
      if(step % checkPointFreq == 0) {
        cout << "Active: current step: " << step;
        cout << " E/N: " << (dpm.getParticleEnergy() + dpm.getParticleKineticEnergy()) / numParticles;
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ) << endl;
        ioDPM.saveParticleActiveConfiguration(outDir);
      }
    }
    if(logSave == true) {
      if(step > (multiple * checkPointFreq)) {
        saveFreq = 10;
        multiple += 1;
      }
      if((step - (multiple-1) * checkPointFreq) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * checkPointFreq) % saveFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveParticleActiveState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveParticleActiveState(currentDir);
      }
    }
    if(step % updateFreq == 0) {
      dpm.calcParticleWallNeighborList(cutDistance);
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioDPM.saveParticleActiveConfiguration(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
