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
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = false, logSave, linSave = false, saveFinal = true;
  long numParticles = 8192, nDim = 2;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), updateFreq = 1;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = 1e03, firstDecade = 0;
  double cutDistance = 1, waveQ, sigma, damping, inertiaOverDamping = atof(argv[6]);
  double ec = 240, timeUnit, timeStep = atof(argv[2]), Tinject = atof(argv[3]), cutoff, maxDelta;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin-fb/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  //dirSample = whichDynamics + "T" + argv[3] + "/iod" + argv[6] + "/";
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
  ioSPFile ioSP(&sp);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      logSave = false;
      outDir = outDir + "dynamics/";
      //outDir = outDir + "dynamics-col/";
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
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setEnergyCostant(ec);
  cutoff = cutDistance * sp.getMinParticleSigma();
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  //timeUnit = sigma * sigma * damping;//epsilon is 1
  timeStep = sp.setTimeStep(timeStep*timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Inertia over damping: " << inertiaOverDamping << ", damping: " << damping << endl;
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  // initialize simulation
  sp.calcParticleBoxNeighborList(cutDistance);
  sp.calcParticleBoxForceEnergy();
  sp.initSoftParticleLangevinFixedBox(Tinject, damping, readState);
  // run integrator
  waveQ = sp.getSoftWaveNumber();
  sp.setInitialPositions();
  while(step != maxStep) {
    sp.softParticleLangevinFixedBoxLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticleEnergy(step, timeStep, waveQ);
      if(step % checkPointFreq == 0) {
        cout << "Brownian: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ) << endl;
        ioSP.saveParticlePacking(outDir);
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
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
