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
#include <stdlib.h>
#include <math.h>
#include <functional>
#include <utility>
#include <thrust/host_vector.h>
#include <experimental/filesystem>

using namespace std;

int main(int argc, char **argv) {
  // variables
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState, logSave, saveLastDecade = true;
  long numParticles = 128, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atoi(argv[3]), updateFreq = 1e02, multiple = 1, saveFreq = 1;
  long checkPointFreq = 1e07, lastFreq = 1e05, saveEnergyFreq = 1e04;
  double cutDistance = 4., timeStep, dt0 = 1.; // relative to the k's
  double ea = 1e03, el = 1, eb = 1e-02, ec = 1;
  // kb/ka should be bound by 1e-04 and 1 and both kc/kl should be 1
  double Epot, temp, phi, Tinject = atof(argv[2]);
  std::string outDir, energyFile, corrFile, currentDir, inDir = argv[1], dirSample = argv[2];
  dirSample = "nve/Tin" + dirSample + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  // read initial configuration
  if (readAndSaveSameDir == true) {
    readState = true;
    inDir = inDir + dirSample;
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
    outDir = inDir;
    if(runDynamics == true) {
      logSave = true;
      outDir = outDir + "dynamics/";
      std::experimental::filesystem::create_directory(outDir);
    }
  } else {
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    if(readAndMakeNewDir == true) {
      readState = true;
      ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
      outDir = inDir + "../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + "nve/") == false) {
        std::experimental::filesystem::create_directory(inDir + "nve/");
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  // output file
  std::experimental::filesystem::create_directory(outDir);
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  timeStep = dpm.setTimeScale(dt0);
  cout << "timeScale: " << timeStep << " Tinject: " << Tinject << endl;
  // initialize simulation
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  dpm.initNVE(Tinject, readState);
  // run NVE integrator
  while(step != maxStep) {
    dpm.NVELoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " potential energy: " << dpm.getPotentialEnergy();
        cout << " T: " << dpm.getTemperature();
        cout << " phi: " << dpm.getPhi() << endl;
      	ioDPM.savePacking(outDir);
        ioDPM.saveState(outDir);
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
        currentDir = outDir + "/t" + std::to_string(step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveTrajectory(currentDir);
      }
    }
    if(saveLastDecade == true) {
      if((step > (9 * checkPointFreq)) && ((step % lastFreq) == 0)) {
        currentDir = outDir + "/t" + std::to_string(step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveTrajectory(currentDir);
      }
    }
    if(step % updateFreq == 0) {
      dpm.calcNeighborList(cutDistance);
    }
    step += 1;
  }
  // save final configuration
  ioDPM.saveConfiguration(outDir);
  ioDPM.closeEnergyFile();

  return 0;
}
