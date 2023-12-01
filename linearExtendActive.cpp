//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/SP2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
#include "include/FIRE.h"
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
  bool readState = true, save = true, saveSame = false, lj = true, wca = false;
  long step, maxStep = atof(argv[8]), printFreq = int(maxStep / 10);
  long numParticles = atol(argv[9]), nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10;
  double phi, pressure, cutoff, maxDelta, strain, Tinject = atof(argv[3]), strainStep = atof(argv[7]);
  double ec = 1, cutDistance = 1, polydispersity = 0.20, sigma, maxStrain = atof(argv[6]), waveQ;
  double Dr = atof(argv[4]), driving = atof(argv[5]), forceUnit;
  std::string inDir = argv[1], outDir, currentDir, energyFile;
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    cutDistance = LJcut-0.5;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
    cutDistance = 1;
  } else {
    cout << "Setting Harmonic potential" << endl;
    cutDistance = 0.5;
    ec = 240;
  }
  ioSPFile ioSP(&sp);
  if(saveSame == true) {
    outDir = inDir;
  } else {
    outDir = inDir + "extend" + argv[6] + "/";
  }
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // save initial configuration
  sp.calcParticleNeighborList(cutDistance);
  ioSP.saveParticlePacking(outDir);
  sp.setEnergyCostant(ec);
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " time step: " << timeStep << endl;
  cout << "Activity - Peclet: " << driving / (damping * Dr * sigma) << " f0: " << driving << " taup: " << 1/Dr << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  Dr = Dr/timeUnit;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, Dr, driving);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  strain = strainStep;
  initBoxSize = sp.getBoxSize();
  while (strain < (maxStrain + strainStep)) {
    //sp.applyExtension(strainStep);
    newBoxSize = initBoxSize;
    newBoxSize[1] *= (1 + strain);
    sp.applyLinearExtension(newBoxSize, strainStep);
    boxSize = sp.getBoxSize();
    cout << "strain: " << strain << " new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Ly0: " << initBoxSize[1] << endl;
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
    // make directory and energy file at each strain step
    std::string currentDir = outDir + "strain" + std::to_string(strain) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "stressEnergy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      ioSP.saveParticleStressEnergy(step, timeStep, numParticles);
      sp.softParticleActiveLangevinLoop();
      if(step % printFreq == 0) {
        cout << "extend Active: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ) << endl;
      }
      maxDelta = sp.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
      }
      step += 1;
    }
    // save minimized configuration
    if(save == true) {
      ioSP.saveParticlePacking(currentDir);
    }
    ioSP.saveParticleStressStrain(strain, numParticles);
    strain += strainStep;
    ioSP.closeEnergyFile();
  }
  if(saveSame == true) {
    ioSP.saveParticlePacking(outDir);
  }
  return 0;
}
