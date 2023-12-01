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
  bool readState = true, save = true, saveSame = false, lj = true;
  long step, maxStep = atof(argv[6]), printFreq = int(maxStep / 10);
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10;
  double phi, pressure, cutoff, maxDelta, strain, Tinject = atof(argv[3]), strainStep = atof(argv[5]);
  double ec = 1, cutDistance = 1, polydispersity = 0.20, sigma, maxStrain = atof(argv[4]), waveQ;
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
  if(saveSame == true) {
    outDir = inDir;
  } else {
    outDir = inDir + "extend" + argv[4] + "/";
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
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << endl;
  cout << "Thermal energy scale: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleLangevin(Tinject, damping, readState);
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
    std::string currentDir = outDir + "strain" + std::to_string(strain) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "stressEnergy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      ioSP.saveParticleStressEnergy(step, timeStep, numParticles);
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
    ioSP.saveParticleStressStrain(strain, numParticles);
    strain += strainStep;
    ioSP.closeEnergyFile();
  }
  if(saveSame == true) {
    ioSP.saveParticlePacking(outDir);
  }

  return 0;
}
