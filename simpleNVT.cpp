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
  bool readState = true, readSame = true, logSave = false, linSave = false, saveFinal = true;
  long numParticles = atol(argv[7]), nDim = 2, maxStep = atof(argv[4]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long step, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0, initialStep = atof(argv[5]);
  double ec = 1, LJcut = 4, cutDistance, cutoff = 1, sigma, damping, Tinject = atof(argv[3]);
  double forceUnit, timeUnit, timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[6]), waveQ, range = 3;
  std::string outDir, energyFile, currentDir, inDir = argv[1];
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
  ioSPFile ioSP(&sp);
  if(logSave == true) {
    outDir = inDir + "simple-log/";
  } else {
    outDir = inDir + "simple/";
  }
  if(readSame == true) {
    if(std::experimental::filesystem::exists(outDir) == true) {
      cout << "reading existing directory" << endl;
      inDir = outDir;
    }
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  std::experimental::filesystem::create_directory(outDir);
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setLJcutoff(LJcut);
  sp.setEnergyCostant(ec);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  damping /= timeUnit;
  ioSP.saveLangevinParams(outDir, sigma, damping);
  // initialize simulation
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  sp.setInitialPositions();
  waveQ = sp.getSoftWaveNumber();
  // range for computing force across fictitious wall
  range *= LJcut * sigma;
  // run integrator
  while(step != maxStep) {
    sp.softParticleLangevinLoop();
    if(step % linFreq == 0) {
      ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
      if(step % checkPointFreq == 0) {
        cout << "Langevin LJ: current step: " << step + initialStep;
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
    }
    if(logSave == true) {
      if(step > (multiple * checkPointFreq)) {
        saveFreq = 1;
        multiple += 1;
      }
      if((step - (multiple-1) * checkPointFreq) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * checkPointFreq) % saveFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
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
