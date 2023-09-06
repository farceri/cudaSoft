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
  long numParticles = 1024, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10);
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), updateFreq = 1e02;
  double cutDistance = 2., damping = 1e03, timeUnit, timeStep = atof(argv[2]);
  double ec = 1, Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]);
  double deltaV, pressure, volume, energy, vscale = atof(argv[7]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = argv[8];
  if(whichDynamics == "langevin/") {
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else if(whichDynamics == "active-langevin/") {
    dirSample = whichDynamics + "Dr" + argv[4] + "/Dr" + argv[4] + "-f0" + argv[5] + "/";
  } else {
    cout << "please specify the correct dynamics" << endl;
  }
  thrust::host_vector<double> boxSize(nDim);
  // initialize sp object
	SP2D sp(numParticles, nDim, numVertexPerParticle);
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
  sigma = sp.getMeanParticleSigma()
  sp.setEnergyCosts(0, 0, 0, ec);
  ioSP.readParticleState(inDir, numParticles, nDim);
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  timeUnit = sigma * sigma * damping;//epsilon is 1
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  Dr = Dr/timeUnit;
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  if(whichDynamics == "langevin/") {
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  } else if(whichDynamics == "active-langevin/") {
    sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  } else {
    cout << "please specify the correct dynamics" << endl;
  }
  // make a volume change by changing the boxSize
  sp.calcParticleForceEnergy();
  pressure = sp.getParticleTotalPressure(driving);
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
    if(whichDynamics == "langevin/") {
      sp.softParticleLangevinLoop();
    } else if(whichDynamics == "active-langevin/") {
      sp.softParticleActiveLangevinLoop();
    } else {
      cout << "please specify the correct dynamics" << endl;
    }
    if(step % saveEnergyFreq == 0) {
      ioSP.saveParticlePressure(step, timeStep, driving);
      if(step % checkPointFreq == 0) {
        cout << "Run: current step: " << step;
        cout << " E: " << sp.getParticleEnergy() + sp.getParticleKineticEnergy();
        cout << " Pdyn: " << sp.getParticleDynamicalPressure();
        cout << " Pactive: " << sp.getParticleActivePressure(driving);
        cout << " T: " << sp.getParticleTemperature() << endl;
      }
    }
    if(step % updateFreq == 0) {
      sp.calcParticleNeighborList(cutDistance);
    }
    step += 1;
  }
  pressure = sp.getParticleTotalPressure(driving);
  cout << "Final pressure: " << pressure << " and energy: " << sp.getParticleEnergy() + sp.getParticleKineticEnergy() << "\n" << endl;

  ioSP.closeEnergyFile();
  ioSP.saveParticleConfiguration(outDir);

  return 0;
}
