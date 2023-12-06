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
  bool readState = true, save = true, lj = true;
  long step, maxStep = atof(argv[6]), printFreq = int(maxStep / 10);
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10;
  double ec = 1, cutDistance = 1, sigma, cutoff, maxDelta, waveQ, Tinject = atof(argv[3]);
  double strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[8]);
  std::string inDir = argv[1], outDir, currentDir, energyFile;
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cutDistance = LJcut-0.5;
    sp.setLJcutoff(LJcut);
  } else {
    ec = 240;
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + "extend" + argv[4] + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + "extend" + argv[4] + "/strain" + std::to_string(initStrain) + "/";
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  } else {
    strain = strainStep;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    initBoxSize = sp.getBoxSize();
  }
  std::experimental::filesystem::create_directory(outDir);
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
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << "Tinject: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  // strain by strainStep up to maxStrain
  while (strain < (maxStrain + strainStep)) {
    newBoxSize = initBoxSize;
    newBoxSize[0] *= (1 + strain);
    sp.applyLinearExtension(newBoxSize, strainStep);
    boxSize = sp.getBoxSize();
    cout << "\nStrain: " << strain << " new box - Ly: " << boxSize[1] << ", Lx: " << boxSize[0] << ", Lx0: " << initBoxSize[0] << endl;
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
    std::string currentDir = outDir + "extend" + std::to_string(strain) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
      sp.softParticleLangevinLoop();
      if(step % printFreq == 0) {
        cout << "extend NVT: current step: " << step;
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
    strain += strainStep;
    ioSP.closeEnergyFile();
  }

  return 0;
}
