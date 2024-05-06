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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave, linSave = false;
  long numParticles = atol(argv[9]), nDim = 2;
  long maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long initialStep = atof(argv[7]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double cutDistance, cutoff = 1, sigma, damping, forceUnit, timeUnit, timeStep = atof(argv[2]), range = 3;
  double ec = 1, Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]), inertiaOverDamping = atof(argv[8]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-wca/";
  dirSample = whichDynamics + "T" + argv[3] + "-tp" + argv[4] + "-f0" + argv[5] + "/";
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  ioSPFile ioSP(&sp);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      //logSave = true;
      //outDir = outDir + "dynamics-log/";
      linSave = true;
      outDir = outDir + "dynamics/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        //if(initialStep != 0) {
        inDir = outDir;
        //}
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
      //outDir = inDir + "../../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleActiveState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setEnergyCostant(ec);
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  //timeUnit = 1 / damping;
  //forceUnit = inertiaOverDamping / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject)*forceUnit << endl;
  cout << "Activity - Peclet: " << driving * tp / (damping * sigma) << " taup: " << tp << " f0: " << driving << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  Dr = 1/(tp*timeUnit);
  ioSP.saveActiveLangevinParams(outDir, sigma, damping, tp, driving);
  // initialize simulation
  sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  sp.setInitialPositions();
  range *= sigma;
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  while(step != maxStep) {
    sp.softParticleActiveLangevinLoop();
    if(step % linFreq == 0) {
      //ioSP.saveParticleSimpleEnergy(step+initialStep, timeStep, numParticles);
      ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
      if(step % checkPointFreq == 0) {
        cout << "Active LJ: current step: " << step + initialStep;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
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
        ioSP.saveParticleNeighbors(currentDir, cutDistance);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleActiveState(currentDir);
        ioSP.saveParticleNeighbors(currentDir, cutDistance);
      }
    }
    step += 1;
  }
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticleActivePacking(outDir);
    ioSP.saveParticleNeighbors(outDir, cutDistance);
  }
  ioSP.closeEnergyFile();

  return 0;
}
