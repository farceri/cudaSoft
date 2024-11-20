//
// Author: Francesco Arceri
// Date:   06-18-2022
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
  bool readState = true, compress = true;
  long numParticles = 1024, nDim = 2;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10);
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), updateFreq = 1e02;
  double cutDistance = 2., damping = 1e03, timeUnit, timeStep = atof(argv[2]);
  double ec = 1, Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]);
  double deltaV, pressure, volume, energy, vscale = atof(argv[7]), cutoff, maxDelta, sigma;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = argv[8];
  thrust::host_vector<double> boxSize(nDim);
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
  outDir = inDir + "comp-delta" + argv[7] + "/";
  if(std::experimental::filesystem::exists(outDir) == false) {
    std::experimental::filesystem::create_directory(outDir);
  } else {
    inDir = outDir;
    compress = false;
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  sigma = sp.getMeanParticleSigma();
  sp.setEnergyCostant(ec);
  cutoff = cutDistance * sp.getMinParticleSigma();
  ioSP.readParticleState(inDir, numParticles, nDim);
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  timeUnit = sigma * sigma * damping;//epsilon is 1
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  Dr = 1/(tp*timeUnit);
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  // make a volume change by changing the boxSize
  sp.calcParticleForceEnergy();
  pressure = sp.getParticlePressure();
  energy = sp.getParticleEnergy() + sp.getParticleKineticEnergy();
  cout << "Initial pressure: " << pressure << " and energy: " << energy << endl;
  if(compress == true) {
    boxSize = sp.getBoxSize();
    volume = boxSize[0] * boxSize[1];
    for (long dim = 0; dim < nDim; dim++) {
      boxSize[dim] *= (1 - vscale);
    }
    sp.setBoxSize(boxSize);
    deltaV = boxSize[0] * boxSize[1] - volume;
  } else {
    cout << "Restarting already compressed configuration" << endl;
    deltaV = 1 - (1 - vscale)*(1 - vscale);
  }
  cout << "Volume change: " << deltaV << endl;
  step = 0;
  while(step != maxStep) {
    sp.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveSimpleEnergy(step, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Run: current step: " << step;
        cout << " E: " << sp.getParticleEnergy() + sp.getParticleKineticEnergy();
        cout << " P: " << sp.getParticlePressure();
        cout << " T: " << sp.getParticleTemperature() << endl;
      }
    }
    maxDelta = sp.getParticleMaxDisplacement();
    if(3*maxDelta > cutoff) {
      sp.calcParticleNeighborList(cutDistance);
      sp.resetLastPositions();
    }
    step += 1;
  }
  pressure = sp.getParticlePressure();
  cout << "Final pressure: " << pressure << " and energy: " << sp.getParticleEnergy() + sp.getParticleKineticEnergy() << "\n" << endl;

  ioSP.closeEnergyFile();
  ioSP.saveParticlePacking(outDir);

  return 0;
}
