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
  bool readState = true, logSave = true, linSave = false, saveFinal = true, constantPressure = true;
  long numParticles = 1024, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), updateFreq = 1e02;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = saveEnergyFreq, firstDecade = 0;
  double cutDistance = 2., waveQ, damping = 1e03, timeUnit, timeStep = atof(argv[2]);
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
  // initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
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
  ioDPM.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(0, 0, 0, ec);
  if(readState == true) {
    ioDPM.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  timeUnit = dpm.getMeanParticleSize()*dpm.getMeanParticleSize()*damping;//epsilon is 1
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  Dr = Dr/timeUnit;
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << " Fext: " << externalForce << endl;
  if(whichDynamics == "active-langevin/") {
    cout << "Velocity Peclet number: " << ((driving/damping) / Dr) / dpm.getMeanParticleSize() << " v0: " << driving / damping << " Dr: " << Dr << endl;
    cout << "Force Peclet number: " << 2. * dpm.getMeanParticleSize() * driving / Tinject << " Tinject: " << Tinject << " driving: " << driving << endl;
  }
  // initialize simulation
  dpm.calcParticleNeighborList(cutDistance);
  dpm.calcParticleForceEnergy();
  if(whichDynamics == "active-langevin/") {
    dpm.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  } else if(whichDynamics == "langevin/") {
    dpm.initSoftParticleLangevin(Tinject, damping, readState);
  }
  // extract external field
  waveQ = dpm.getSoftWaveNumber();
  dpm.makeExternalParticleForce(externalForce);
  ioDPM.save1DFile(outDir + "/externalField.dat", dpm.getExternalParticleForce());
  while(step != maxStep) {
    if(constantPressure == true) {
      pscale = 1 + (timeStep / 3 * taup) * beta * (dpm.getParticleTotalPressure(driving) - p0);
    }
    if(whichDynamics == "active-langevin/") {
      dpm.softParticleALExtFieldLoop();
    } else if(whichDynamics == "langevin/") {
      dpm.softParticleLExtFieldLoop();
    }
    if(constantPressure == true) {
      dpm.pressureScaleParticles(pscale);
    }
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveParticleEnergy(step, timeStep, waveQ);
      if(step % checkPointFreq == 0) {
        cout << "FDT: current step: " << step;
        cout << " Fmax: " << dpm.getParticleMaxUnbalancedForce();
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ) << endl;
        //ioDPM.saveParticleConfiguration(outDir);
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
        ioDPM.saveParticleConfiguration(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveParticleConfiguration(currentDir);
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
