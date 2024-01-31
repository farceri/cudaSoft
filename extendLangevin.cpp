//
// Author: Francesco Arceri
// Date:   10-03-2021
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
  bool readState = true, compress = false, biaxial = true, lj = true, wca = false;
  bool saveFinal = true, logSave = false, linSave = true, savePacking = true;
  long numParticles = atol(argv[7]), nDim = 2;
  long maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, Tinject = atof(argv[3]), cutoff, LJcut = 5.5, sigma, timeUnit, timeStep = atof(argv[2]), waveQ;
  double cutDistance, damping, inertiaOverDamping = atof(argv[6]), strain=atof(argv[8]), strainx, sign = 1, range;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    sign = -1;
    dirSample = "compress";
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
  } else {
    cout << "Setting Harmonic potential" << endl;
    cutDistance = 0.5;
    ec = 240;
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + dirSample + argv[8] + "/";//+ "-" + argv[9]
  if(initialStep != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    inDir = outDir;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  } else {
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    initBoxSize = sp.getBoxSize();
  }
  std::experimental::filesystem::create_directory(outDir);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setEnergyCostant(ec);
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  damping /= timeUnit;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, 0, 0);
  if(initialStep == 0) {
    strainx = -strain / (1 + strain);
    boxSize = sp.getBoxSize();
    boxSize[1] *= (1 + sign * strain);
    if(biaxial == true) {
      strainx = -strain/(1 + strain);
      boxSize[0] *= (1 + sign * strainx);
      cout << "strainx: " << strainx << endl;
      sp.applyBiaxialExtension(boxSize, sign * strain, sign * strainx);
    } else {
      sp.applyLinearExtension(boxSize, sign * strain);
    }
    boxSize = sp.getBoxSize();
    cout << "strain: " << strain << ", density: " << sp.getParticlePhi() << endl;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Abox: " << boxSize[0]*boxSize[1] << endl;
    cout << "old box - Lx0: " << initBoxSize[0] << ", Ly0: " << initBoxSize[1] << ", Abox0: " << initBoxSize[0]*initBoxSize[1] << endl;
  } else {
    cout << "restarting configuration - energy.dat will be overridden" << endl;
  }
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  sp.setDisplacementCutoff(cutoff, cutDistance);
  sp.resetUpdateCount();
  waveQ = sp.getSoftWaveNumber();
  sp.setInitialPositions();
  // range for computing force across fictitious wall
  range = 5 * LJcut * sigma;
  // run integrator
  while(step != maxStep) {
    sp.softParticleLangevinLoop();
    if(step % checkPointFreq == 0) {
      cout << "Extend NVT: current step: " << step + initialStep;
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
      if(saveFinal == true) {
        ioSP.saveParticlePacking(outDir);
      }
    }
    if(logSave == true) {
      if(step > (multiple * maxStep)) {
        saveFreq = 1;
        multiple += 1;
      }
      if((step - (multiple-1) * maxStep) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * maxStep) % saveFreq) == 0) {
        ioSP.saveParticleWallEnergy(step + initialStep, timeStep, numParticles, range);
        if(savePacking == true) {
          currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
          std::experimental::filesystem::create_directory(currentDir);
          ioSP.saveParticleState(currentDir);
          ioSP.saveParticleNeighbors(currentDir);
        }
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        ioSP.saveParticleWallEnergy(step + initialStep, timeStep, numParticles, range);
        if(savePacking == true) {
          currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
          std::experimental::filesystem::create_directory(currentDir);
          ioSP.saveParticleState(currentDir);
          ioSP.saveParticleNeighbors(currentDir);
        }
      }
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
