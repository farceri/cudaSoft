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
  bool readState = true, logSave = true, linSave = true, saveFinal = true;
  long numParticles = 1024, nDim = 2;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10);
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = saveEnergyFreq, firstDecade = 0;
  double cutDistance, cutoff = 2, damping = 1e03, timeUnit, timeStep = atof(argv[2]);
  double ec = 240, Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = argv[9];
  double p0 = atof(argv[8]), pscale, beta = 1, taup = 1e-02;
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(whichDynamics == "langevin/") {
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else if(whichDynamics == "active-langevin/") {
    dirSample = whichDynamics + "/tp" + argv[4] + "-f0" + argv[5] + "/";
    sp.setParticleType(simControlStruct::particleEnum::active);
    sp.setSelfPropulsionParams(driving, tp);
  } else {
    cout << "please specify the correct dynamics" << endl;
  }
  ioSPFile ioSP(&sp);
  // set input and output
  inDir = inDir + dirSample;
  outDir = inDir + "dynamics-ptot" + argv[8] + "/";
  if(std::experimental::filesystem::exists(outDir) == true) {
    inDir = outDir;
    initialStep = atof(argv[7]);
  } else {
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  sp.setEnergyCostant(ec);
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
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << " p0: " << p0 << endl;
  if(whichDynamics == "active-langevin/") {
    cout << "Velocity Peclet number: " << ((driving/damping) / Dr) / sp.getMeanParticleSigma() << " v0: " << driving / damping << " Dr: " << Dr << endl;
    cout << "Force Peclet number: " << sp.getMeanParticleSigma() * driving / Tinject << " Tinject: " << Tinject << " driving: " << driving << endl;
    //taup = 1/Dr;
  }
  // initialize simulation
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  //sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  while(step != maxStep) {
    pscale = 1 + (timeStep / 3 * taup) * beta * (sp.getParticlePressure() - p0);
    //pscale = 1 + (timeStep / 3 * taup) * beta * (sp.getParticleDynamicalPressure() - p0);
    sp.softParticleLangevinLoop();
    sp.pressureScaleParticles(pscale);
    if(step % saveEnergyFreq == 0) {
      ioSP.saveSimpleEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "NPT: current step: " << step;
        cout << " Ptot: " << sp.getParticlePressure();
        cout << " T: " << sp.getParticleTemperature();
        cout << " phi: " << sp.getParticlePhi() << endl;
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
        ioSP.saveParticlePacking(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticlePacking(currentDir);
      }
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
