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
  bool readState = true, lj = true, wca = false, compress = false, biaxial = true, centered = false;
  bool saveFinal = true, logSave = true, linSave = false, savePacking = false;
  long numParticles = atol(argv[9]), nDim = 2;
  long maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long initialStep = atof(argv[7]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, cutDistance, LJcut = 4, cutoff, sigma, damping, forceUnit, timeUnit, sign = 1, range, waveQ;
  double timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[8]), strain = atof(argv[10]), strainx;
  double Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
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
  outDir = inDir + dirSample + argv[10] + "/";//+ "-" + argv[11]
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
    ioSP.readParticleActiveState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // save initial configuration
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
  if(initialStep == 0) {
    strainx = -strain / (1 + strain);
    boxSize = sp.getBoxSize();
    boxSize[1] *= (1 + sign * strain);
    if(biaxial == true) {
      boxSize[0] *= (1 + sign * strainx);
      cout << "strainx: " << strainx << endl;
      if(centered == true) {
        sp.applyCenteredBiaxialExtension(boxSize, sign * strain, sign * strainx);
      } else {
        sp.applyBiaxialExtension(boxSize, sign * strain, sign * strainx);
      }
    } else {
      if(centered == true) {
        sp.applyCenteredUniaxialExtension(boxSize, sign * strain, direction);
      } else {
        sp.applyUniaxialExtension(boxSize, sign * strain, direction);
      }
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
  sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  cutoff = 2 * (1 + cutDistance) * sp.getMinParticleSigma();
  sp.setDisplacementCutoff(cutoff, cutDistance);
  sp.resetUpdateCount();
  waveQ = sp.getSoftWaveNumber();
  sp.setInitialPositions();
  // range for computing force across fictitious wall
  range = 2.5 * LJcut * sigma;
  // run integrator
  while(step != maxStep) {
    sp.softParticleActiveLangevinLoop();
    if(step % checkPointFreq == 0) {
      cout << "Extend Active: current step: " << step + initialStep;
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
        ioSP.saveParticleActivePacking(outDir);
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
          ioSP.saveParticleActiveState(currentDir);
        }
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        ioSP.saveParticleWallEnergy(step + initialStep, timeStep, numParticles, range);
        if(savePacking == true) {
          currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
          std::experimental::filesystem::create_directory(currentDir);
          ioSP.saveParticleActiveState(currentDir);
        }
      }
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticleActivePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
