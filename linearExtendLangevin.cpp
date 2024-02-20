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
  bool readState = true, save = true, saveSame = false, lj = false, adh = true, wca = false, compress = true, biaxial = true;
  long step, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100), direction = 0;
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, updateCount = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 5.5, damping, inertiaOverDamping = 10, strainx, strainStepx;
  double ec = 1, cutDistance = 1, sigma, cutoff, maxDelta, waveQ, Tinject = atof(argv[3]), sign = 1, range = 3;
  double l1 = pow(2, 1/6), l2 = 3.3, strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[8]);
  std::string inDir = argv[1], outDir, currentDir, energyFile, dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    if(biaxial == true) {
      sign = -1;
      dirSample = "biaxial-comp";
    }
  } else if(biaxial == true) {
    dirSample = "biaxial";
  }
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    cutDistance = LJcut+0.5;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
    cutDistance = 1;
  } else if(adh == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::adhesive);
    cout << "Setting Adhesive potential" << endl;
    cutDistance = l2+0.5;
    sp.setAdhesionParams(l1, l2);
  } else {
    cout << "Setting Harmonic potential" << endl;
    cutDistance = 0.5;
    ec = 240;
  }
  ioSPFile ioSP(&sp);
  //outDir = inDir + dirSample + argv[5] + "-tmax" + argv[6] + "/";
  outDir = inDir + dirSample + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    outDir = inDir + dirSample + "/";
    inDir = inDir + dirSample + "/strain" + argv[8] + "/";
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
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  // strain by strainStep up to maxStrain
  strainStepx = -strainStep / (1 + strainStep);
  while (strain < (maxStrain + strainStep)) {
    newBoxSize = initBoxSize;
    if(biaxial == true) {
      newBoxSize[1] *= (1 + sign * strain);
      strainx = -strain/(1 + strain);
      newBoxSize[0] *= (1 + sign * strainx);
      cout << "strainx: " << strainx << endl;
      sp.applyBiaxialExtension(newBoxSize, sign * strainStep, sign * strainStepx);
    } else {
      newBoxSize[direction] *= (1 + sign * strain);
      sp.applyLinearExtension(newBoxSize, sign * strainStep, direction);
    }
    boxSize = sp.getBoxSize();
    cout << "strain: " << strain << ", density: " << sp.getParticlePhi() << endl;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Abox: " << boxSize[0]*boxSize[1] << endl;
    cout << "old box - Lx0: " << initBoxSize[0] << ", Ly0: " << initBoxSize[1] << ", Abox0: " << initBoxSize[0]*initBoxSize[1] << endl;
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
    sp.setDisplacementCutoff(cutoff, cutDistance);
    sp.resetUpdateCount();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
    // range for computing force across fictitious wall
    range *= LJcut * sigma;
    std::string currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      if(step % linFreq == 0) {
        ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
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
