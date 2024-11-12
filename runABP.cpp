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
  // read input and make new directory: everything false
  // read and save same directory: readAndSaveSameDir = true
  // read directory and save in new directory: readAndMakeNewDir = true
  // read directory and save in "dynamics" dirctory: readAndSaveSameDir = true and runDynamics = true
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  bool fixedbc = false, roundbc = true, reflect = false, reflectnoise = false;
  bool initAngles = false, readState = true, saveFinal = true, logSave = false, linSave = true;
  // variables
  long maxStep = atof(argv[4]), initialStep = atof(argv[5]), numParticles = atol(argv[6]), nDim = 2;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, timeStep = atof(argv[2]), alphaUnit, timeUnit, velUnit, sigma, LJcut = 4, cutDistance, cutoff = 0.5, waveQ;
  double ew = 10*ec, Tinject = 0, driving = 2, damping = 1, tp = atof(argv[3]);
  std::string inDir = argv[1], outDir, currentDir, dirSample, potType = argv[7], energyFile, whichDynamics = "active/";
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setParticleType(simControlStruct::particleEnum::active);
  sp.setNoiseType(simControlStruct::noiseEnum::activeNoise);
  if(numParticles < 256) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
    sp.setBoxEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundBox);
    sp.setBoxEnergyScale(ew);
  } else {
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
  }
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  if(reflect == true) {
    whichDynamics = "active/smooth/";
    sp.setBoxType(simControlStruct::boxEnum::reflect);
  } else if(reflectnoise == true) {
    whichDynamics = "active/noise/";
    sp.setBoxType(simControlStruct::boxEnum::reflectnoise);
  } else {
    whichDynamics = "active/wall/";
  }
  dirSample = whichDynamics + "tp" + argv[3] + "/";
  // set input and output
  ioSPFile ioSP(&sp);
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics";
      if(logSave == true) {
        outDir = outDir + "-log/";
      } else {
        outDir = outDir + "/";
      }
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
    } else {
      initAngles = true; // initializing from NVT
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    if(initAngles == true) {
      ioSP.readParticleVelocity(inDir, numParticles, nDim);
      sp.initializeParticleAngles();
    } else {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ec);
  velUnit = sigma / timeUnit;
  alphaUnit = ec / (sigma * sigma);
  driving = sqrt(2*damping*driving) / damping;
  cout << "Units - time: " << timeUnit << " space: " << sigma << " velocity: " << velUnit << " time step: " << timeStep << endl;
  cout << "Noise - damping: " << damping << " driving: " << driving << " taup: " << tp << " magnitude: " << sqrt(2 * timeStep / tp) << endl;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  tp *= timeUnit;
  driving *= velUnit;
  damping /= timeUnit;
  sp.setSelfPropulsionParams(driving, tp);
  ioSP.saveLangevinParams(outDir, damping);
  // initialize simulation
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  ioSP.saveParticlePacking(outDir);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  waveQ = sp.getSoftWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  while(step != maxStep) {
    sp.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveAlignEnergy(step+initialStep, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Vicsek: current step: " << step + initialStep;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
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
          //ioSP.saveParticleNeighbors(outDir);
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
        //ioSP.saveParticleNeighbors(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
        //ioSP.saveParticleNeighbors(currentDir);
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
    ioSP.saveParticlePacking(outDir);
    //ioSP.saveParticleNeighbors(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}