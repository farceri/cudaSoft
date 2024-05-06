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
  bool readState = false, save = true, saveSame = false, lj = true;
  long step, maxStep = 1e04, printFreq = int(maxStep / 10);
  long numParticles = atol(argv[5]), nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10;
  double phi, pressure, maxDelta, strain, Tinject = atof(argv[6]), strainStep = atof(argv[4]);
  double ec = 1, cutDistance, cutoff = 1, polydispersity = 0.20, sigma, maxStrain = atof(argv[3]);
  std::string inDir = argv[1], outDir, currentDir, dirSample = "shear-NVT/", energyFile;
	// initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cutDistance = LJcut+0.5;
    dirSample = "shear-NVTLJ-5.5/";
  }
  ioSPFile ioSP(&sp);
  if(saveSame == true) {
    outDir = inDir;
  } else {
    outDir = inDir + dirSample;
  }
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  // output file
  energyFile = outDir + "stressEnergy.dat";
  ioSP.openEnergyFile(energyFile);
  sp.setEnergyCostant(ec);
  sp.setLJcutoff(LJcut);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << endl;
  cout << "Thermal energy scale: " << Tinject << endl;
  ioSP.saveLangevinParams(outDir, sigma, damping);
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  strain = strainStep;
  while (strain < (maxStrain + strainStep)) {
    sp.setLEshift(strain);
    sp.applyLEShear(strainStep);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    step = 0;
    while(step != maxStep) {
      sp.softParticleLangevinLoop();
      if(step % printFreq == 0) {
        cout << "shear NVT: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature() << endl;
      }
      step += 1;
    }
    // save minimized configuration
    if(save == true) {
      std::string currentDir = outDir + "strain" + std::to_string(strain) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioSP.saveParticlePacking(currentDir);
    }
    ioSP.saveParticleStressStrain(strain, numParticles);
    strain += strainStep;
  }
  if(saveSame == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
