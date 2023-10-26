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
  bool readState = false, save = true, saveSame = false;
  long step, maxStep = 1e04, printFreq = int(maxStep / 10);
  long numParticles = 8192, nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double timeStep = 1e-02, timeUnit, LJcut = 2.5, damping, inertiaOverDamping = 10;
  double phi, pressure, cutoff, maxDelta, strain, Tinject = 1e-03, strainStep = atof(argv[3]);
  double ec = 1, cutDistance = LJcut-0.5, polydispersity = 0.20, sigma, maxStrain = atof(argv[2]);
  std::string inDir = argv[1], outDir, currentDir, dirSample = "shear-NVTLJ/", saveFile;
	// initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
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
  saveFile = outDir + "output.dat";
  ioSP.openOutputFile(saveFile);
  sp.setEnergyCostant(ec);
  sp.setLJcutoff(LJcut);
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << endl;
  cout << "Thermal energy scale: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
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
        cout << " T: " << sp.getParticleTemperature();
        cout << " pressure: " << sp.getParticleVirialPressure();
        cout << " shear stress: " << sp.getParticleShearStress() << endl;
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
      std::string currentDir = outDir + "strain" + std::to_string(strain) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioSP.saveAthermalParticlePacking(currentDir);
    }
    ioSP.saveParticleStress(strain, numParticles);
    strain += strainStep;
  }

  return 0;
}
