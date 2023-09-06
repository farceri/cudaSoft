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
  bool readState = true, logSave = true, linSave = false, saveFinal = true, constantPressure = true;
  long numParticles = 1024, nDim = 2;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), updateFreq = 1e02;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = saveEnergyFreq, firstDecade = 0;
  double cutDistance = 2., waveQ, damping = 1e03, timeUnit, timeStep = atof(argv[2]), cutoff, maxDelta;
  double ec = 240, Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-langevin/";
  double externalForce = atof(argv[8]), p0 = atof(argv[9]), pscale, beta = 1, taup = 1e-02;
  if(whichDynamics == "active-langevin/") {
    dirSample = whichDynamics + "Dr" + argv[4] + "/Dr" + argv[4] + "-f0" + argv[5] + "/";
  } else if(whichDynamics == "langevin/") {
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else {
    step = maxStep;
    cout << "Please specify the dynamics you want to run" << endl;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  // set input and output
  inDir = inDir + dirSample;
  if(constantPressure == false) {
    outDir = inDir + "dynamics-fext" + argv[8] + "/";
  } else {
    outDir = inDir + "dynamics-fext" + argv[8] + "-p0" + argv[9] + "/";
  }
  if(std::experimental::filesystem::exists(outDir) == true) {
    inDir = outDir;
    initialStep = atof(argv[7]);
  } else {
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  sp.setEnergyCostant(ec);
  cutoff = cutDistance * sp.getMinParticleSigma();
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  timeUnit = sp.getMeanParticleSigma()*sp.getMeanParticleSigma()*damping;//epsilon is 1
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  Dr = Dr/timeUnit;
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << " Fext: " << externalForce << endl;
  if(whichDynamics == "active-langevin/") {
    cout << "Velocity Peclet number: " << ((driving/damping) / Dr) / sp.getMeanParticleSigma() << " v0: " << driving / damping << " Dr: " << Dr << endl;
    cout << "Force Peclet number: " << 2. * sp.getMeanParticleSigma() * driving / Tinject << " Tinject: " << Tinject << " driving: " << driving << endl;
  }
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  if(whichDynamics == "active-langevin/") {
    sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  } else if(whichDynamics == "langevin/") {
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  }
  // extract external field
  waveQ = sp.getSoftWaveNumber();
  sp.makeExternalParticleForce(externalForce);
  ioSP.save1DFile(outDir + "/externalField.dat", sp.getExternalParticleForce());
  while(step != maxStep) {
    if(constantPressure == true) {
      pscale = 1 + (timeStep / 3 * taup) * beta * (sp.getParticleTotalPressure(driving) - p0);
    }
    if(whichDynamics == "active-langevin/") {
      sp.softParticleActiveExtFieldLoop();
    } else if(whichDynamics == "langevin/") {
      sp.softParticleLangevinExtFieldLoop();
    }
    if(constantPressure == true) {
      sp.pressureScaleParticles(pscale);
    }
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticleEnergy(step, timeStep, waveQ);
      if(step % checkPointFreq == 0) {
        cout << "FDT: current step: " << step;
        cout << " Fmax: " << sp.getParticleMaxUnbalancedForce();
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ) << endl;
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
