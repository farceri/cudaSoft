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
  bool readState = true, save = true, saveSame = false, lj = true, wca = false, compress = false;
  long step, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, updateCount = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10;
  double ec = 1, cutDistance = 1, sigma, cutoff, maxDelta, waveQ, Tinject = atof(argv[3]), sign = 1;
  double strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[8]);
  std::string inDir = argv[1], outDir, currentDir, energyFile, dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    sign = -1;
    dirSample = "compress";
  }
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
  outDir = inDir + dirSample + argv[5] + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + dirSample + argv[5] + "/strain" + std::to_string(initStrain) + "/";
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
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << "Tinject: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  // strain by strainStep up to maxStrain
  while (strain < (maxStrain + strainStep)) {
    newBoxSize = initBoxSize;
    newBoxSize[0] *= (1 + sign * strain);
    sp.applyLinearExtension(newBoxSize, sign * strainStep);
    boxSize = sp.getBoxSize();
    cout << "\nStrain: " << strain << " new box - Ly: " << boxSize[1] << ", Lx: " << boxSize[0] << ", Lx0: " << initBoxSize[0] << endl;
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    cutoff = (1 + cutDistance) * sigma;
    sp.setDisplacementCutoff(cutoff, cutDistance);
    sp.resetUpdateCount();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
    std::string currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      if(step % linFreq == 0) {
        ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
      }
      sp.softParticleLangevinLoop();
      if(step % checkPointFreq == 0) {
        cout << "Extend Langevin: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        sp.resetUpdateCount();
        if(save == true) {
          ioSP.saveParticlePacking(currentDir);
        }
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
  if(saveSame == true) {
    ioSP.saveParticlePacking(outDir);
  }
  return 0;
}
