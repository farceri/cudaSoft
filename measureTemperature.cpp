//
// Author: Francesco Arceri
// Date:   08-16-2022
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
  bool readState = true, zeroOutMassiveVel = true;
  bool logSave = false, linSave = true, saveFinal = true;
  long numParticles = 1024, nDim = 2;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), updateFreq = 1e01;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = int(saveEnergyFreq / 10), firstDecade = 0, firstIndex = 10;
  double cutDistance = 2., damping = 1e03, timeUnit, timeStep = atof(argv[2]), maxDelta, cutoff;
  double ec = 1, sigma, Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-langevin/";
  double mass = atof(argv[8]);
  if(whichDynamics == "langevin/") {
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else if(whichDynamics == "active-langevin/") {
    dirSample = whichDynamics + "/Dr" + argv[4] + "-f0" + argv[5] + "/T" + argv[3] + "/";
  } else {
    cout << "please specify the correct dynamics" << endl;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  // set input and output
  inDir = inDir + dirSample;
  outDir = inDir + "dynamics-mass" + argv[8] + "/";
  if(std::experimental::filesystem::exists(outDir) == true) {
    inDir = outDir;
    initialStep = atof(argv[7]);
    zeroOutMassiveVel = false;
  } else {
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  sigma = sp.getMeanParticleSigma();
  sp.setEnergyCostant(ec);
  cutoff = cutDistance * sp.getMinParticleSigma();
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  timeUnit = sigma * sigma * damping;//epsilon is 1
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  Dr = Dr/timeUnit;
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << endl;
  if(whichDynamics == "active-langevin/") {
    cout << "Velocity Peclet number: " << ((driving/damping) / Dr) / sigma << " v0: " << driving / damping << " Dr: " << Dr << endl;
    cout << "Force Peclet number: " << 2. * sigma * driving / Tinject << " Tinject: " << Tinject << " driving: " << driving << endl;
  }
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  if(whichDynamics == "langevin/") {
    sp.initSoftParticleLangevinSubSet(Tinject, damping, firstIndex, mass, readState, zeroOutMassiveVel);
  } else if(whichDynamics == "active-langevin/") {
    sp.initSoftParticleActiveSubSet(Tinject, Dr, driving, damping, firstIndex, mass, readState, zeroOutMassiveVel);
  } else {
    cout << "please specify the correct dynamics" << endl;
  }
  while(step != maxStep) {
    if(whichDynamics == "langevin/") {
      sp.softParticleLangevinSubSetLoop();
    } else if(whichDynamics == "active-langevin/") {
      sp.softParticleActiveSubSetLoop();
    } else {
      cout << "please specify the correct dynamics" << endl;
    }
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticleSimpleEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "MP: current step: " << step;
        cout << " T: " << sp.getParticleTemperature();
        cout << " Tmassive: " << sp.getMassiveTemperature(firstIndex, mass) << endl;
        //ioSP.saveParticleConfiguration(outDir);
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
        ioSP.saveParticleConfiguration(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleConfiguration(currentDir);
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
    ioSP.saveParticleConfiguration(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
