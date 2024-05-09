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
  bool readState = true, save = true, lj = true, adh = false, wca = false, compress = false, biaxial = true, centered = false;
  long step, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, updateCount = 0, direction = 0;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, damping, inertiaOverDamping = 10, strainx, strainStepx;
  double ec = 1, cutDistance, cutoff = 0.5, sigma,  waveQ, Tinject = atof(argv[3]), sign = 1, range = 3, prevEnergy = 0;
  double l1 = pow(2, 1/6), l2 = 3.3, strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[8]);
  std::string inDir = argv[1], outDir, currentDir, timeDir, energyFile, dirSample = "nvt-ext";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
  SP2D sp(numParticles, nDim);
  if(compress == true) {
    sign = -1;
    if(biaxial == true) {
      dirSample = "nvt-biaxial-comp";
    } else {
      dirSample = "nvt-comp";
    }
  } else if(biaxial == true) {
    dirSample = "nvt-biaxial-ext";
  }
  if(centered == true) {
    dirSample = dirSample + "-centered";
  }
  sp.setEnergyCostant(ec);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
  } else if(adh == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::adhesive);
    cout << "Setting Adhesive potential" << endl;
    sp.setAdhesionParams(l1, l2);
  } else {
    cout << "Setting Harmonic potential" << endl;
    cutoff = 1;
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + dirSample + argv[5] + "-tmax" + argv[6] + "/";
  //outDir = inDir + dirSample + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + dirSample + argv[5] + "-tmax" + argv[6] + "/strain" + argv[8] + "/";
    //inDir = inDir + dirSample + "/strain" + argv[8] + "/";
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
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  ioSP.saveLangevinParams(outDir, sigma, damping);
  range *= LJcut * sigma;
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  waveQ = sp.getSoftWaveNumber();
  // strain by strainStep up to maxStrain
  strainStepx = -sign * strainStep / (1 + sign * strainStep);
  while (strain < (maxStrain + strainStep)) {
    if(biaxial == true) {
      newBoxSize[1] = (1 + sign * strain) * initBoxSize[1];
      strainx = -sign * strain / (1 + sign * strain);
      newBoxSize[0] = (1 + sign * strainx) * initBoxSize[0];
      cout << "strainx: " << strainx << endl;
      if(centered == true) {
        sp.applyCenteredBiaxialExtension(newBoxSize, sign * strainStep, strainStepx);
      } else {
        sp.applyBiaxialExtension(newBoxSize, sign * strainStep, strainStepx);
      }
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + sign * strain) * initBoxSize[direction];
      if(centered == true) {
        sp.applyCenteredUniaxialExtension(newBoxSize, sign * strainStep, direction);
      } else {
        sp.applyUniaxialExtension(newBoxSize, sign * strainStep, direction);
      }
    }
    boxSize = sp.getBoxSize();
    cout << "strain: " << sign * strain << ", density: " << sp.getParticlePhi() << endl;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Abox: " << boxSize[0]*boxSize[1] << endl;
    cout << "old box - Lx0: " << initBoxSize[0] << ", Ly0: " << initBoxSize[1] << ", Abox0: " << initBoxSize[0]*initBoxSize[1] << endl;
    currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
    sp.resetUpdateCount();
    step = 0;
    sp.setInitialPositions();
    while(step != maxStep) {
      if(step % linFreq == 0) {
        ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
      }
      sp.softParticleLangevinLoop();
      if(step % checkPointFreq == 0) {
        cout << "Extend NVT: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        //cout << " F: " << sp.getParticleWallForce(range);
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
          //timeDir = currentDir + "/t" + std::to_string(step) + "/";
          //std::experimental::filesystem::create_directory(timeDir);
          //ioSP.saveParticlePacking(timeDir);
          //ioSP.saveParticleNeighbors(timeDir);
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
  return 0;
}