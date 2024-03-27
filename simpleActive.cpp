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
  long numParticles = atol(argv[9]), nDim = 2, maxStep = atof(argv[6]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long step, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0, initialStep = atof(argv[7]);
  double ec = 1, LJcut = 4, cutDistance, cutoff = 1, sigma, damping, range = 3;
  double forceUnit, timeUnit, timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[8]);
  double Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]), waveQ;
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
    ioSP.readParticleActiveState(inDir, numParticles, nDim);
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
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  cout << "Activity - Peclet: " << driving * tp / (damping * sigma) << " taup: " << tp << " f0: " << driving << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  Dr = 1/(tp * timeUnit);
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, Dr, driving);
  // initialize simulation
  sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
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
    sp.softParticleActiveLangevinLoop();
    if(step % linFreq == 0) {
      ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
      if(step % checkPointFreq == 0) {
        cout << "Active LJ: current step: " << step + initialStep;
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
        ioSP.saveParticleActiveState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleActiveState(currentDir);
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
