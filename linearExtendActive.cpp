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
  bool readState = true, save = true, lj = true, wca = false, compress = false, biaxial = true, centered = false;
  long step, maxStep = atof(argv[8]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long numParticles = atol(argv[9]), nDim = 2, minStep = 20, numStep = 0, updateCount = 0, direction = 1;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, damping, inertiaOverDamping = 10, strain, initStrain = atof(argv[10]);
  double ec = 1, cutDistance = 1, maxStrain = atof(argv[6]), strainStep = atof(argv[7]), strainx, strainStepx, range = 3;
  double cutoff, sigma, forceUnit, waveQ, Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]), sign = 1;
  std::string inDir = argv[1], outDir, currentDir, energyFile, dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    sign = -1;
    if(biaxial == true) {
      dirSample = "biaxial-comp";
    } else {
      dirSample = "compress";
    }
  } else if(biaxial == true) {
    dirSample = "biaxial";
  }
  if(centered == true) {
    dirSample = dirSample + "-centered";
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
  } else {
    cout << "Setting Harmonic potential" << endl;
    cutDistance = 0.5;
    ec = 240;
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + dirSample + argv[7] + "-tmax" + argv[8] + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + dirSample + argv[7] + "-tmax" + argv[8] + "/strain" + std::to_string(initStrain) + "/";
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  } else {
    strain = strainStep;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    initBoxSize = sp.getBoxSize();
  }
  std::experimental::filesystem::create_directory(outDir);
  if(readState == true) {
    ioSP.readParticleActiveState(inDir, numParticles, nDim);
  }
  // save initial configuration
  ioSP.saveParticleActivePacking(outDir);
  sp.setEnergyCostant(ec);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  cout << "Activity - Peclet: " << driving * tp / (damping * sigma) << " taup: " << tp << " f0: " << driving << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  Dr = 1/(tp * timeUnit);
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, Dr, driving);
  range *= LJcut * sigma;
  // strain by strainStep up to maxStrain
  strainStepx = -strainStep / (1 + strainStep);
  while (strain < (maxStrain + strainStep)) {
    if(biaxial == true) {
      newBoxSize[1] = (1 + sign * strain) * initBoxSize[1];
      strainx = -strain / (1 + strain);
      newBoxSize[0] = (1 + sign * strainx) * initBoxSize[0];
      cout << "strainx: " << strainx << endl;
      if(centered == true) {
        sp.applyCenteredBiaxialExtension(newBoxSize, sign * strainStep, sign * strainStepx);
      } else {
        sp.applyBiaxialExtension(newBoxSize, sign * strainStep, sign * strainStepx);
      }
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + sign * strain) * initBoxSize[direction];
      if(centered == true) {
        sp.applyCenteredUniaxialExtension(boxSize, sign * strainStep, direction);
      } else {
        sp.applyUniaxialExtension(boxSize, sign * strainStep, direction);
      }
    }
    boxSize = sp.getBoxSize();
    cout << "strain: " << strain << ", density: " << sp.getParticlePhi() << endl;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Abox: " << boxSize[0]*boxSize[1] << endl;
    cout << "old box - Lx0: " << initBoxSize[0] << ", Ly0: " << initBoxSize[1] << ", Abox0: " << initBoxSize[0]*initBoxSize[1] << endl;
    sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);  
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    cutoff = 2 * (1 + cutDistance) * sp.getMinParticleSigma();
    sp.setDisplacementCutoff(cutoff, cutDistance);
    sp.resetUpdateCount();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
    // range for computing force across fictitious wall
    std::string currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    while(step != maxStep) {
      if(step % linFreq == 0) {
        ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
      }
      sp.softParticleActiveLangevinLoop();
      if(step % checkPointFreq == 0) {
        cout << "Extend Active: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " F: " << sp.getParticleWallForce(range);
        cout << " ISF: " << sp.getParticleISF(waveQ);
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        sp.resetUpdateCount();
        if(save == true) {
          ioSP.saveParticleActivePacking(currentDir);
        }
      }
      step += 1;
    }
    // save minimized configuration
    if(save == true) {
      ioSP.saveParticleActivePacking(currentDir);
    }
    strain += strainStep;
    ioSP.closeEnergyFile();
  }
  return 0;
}
