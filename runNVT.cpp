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
  // read input and make new directory denoted by T: everything false
  // read and save same directory denoted by T: readAndSaveSameDir = true
  // read directory denoted by T and save in new directory denoted by T: readAndMakeNewDir = true
  // read directory denoted by T and save in "dynamics" dirctory: readAndSaveSameDir = true and runDynamics = true
  // read NH directory denoted by T for all previous options: readNH = true
  // save in "damping" directory for all the previous options: dampingDir = true
  // read input and save in "dynamics" directory: justRun = true
  bool readNH = false, dampingDir = true, justRun = false;
  bool readAndMakeNewDir = true, readAndSaveSameDir = false, runDynamics = false;
  // variables
  bool fixedbc = false, roundbc = true, fixedSides = false, reflect = false;
  bool readState = true, saveFinal = true, logSave = false, linSave = true;
  long numParticles = atol(argv[7]), nDim = atol(argv[8]), maxStep = atof(argv[4]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = atof(argv[10]), ew = ec, LJcut = 4, cutDistance, cutoff = 0.5, waveQ, timeStep = atof(argv[2]);
  double Tinject = atof(argv[3]), damping, inertiaOverDamping = atof(argv[6]), sigma, forceUnit, timeUnit, range = 3;
  std::string outDir, energyFile, currentDir, inDir = argv[1], potType = argv[9], dirSample, whichDynamics = "langevin";
  if(nDim == 3) {
    LJcut = 2.5;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
  if(numParticles < 256) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  if(readNH == true) {
    whichDynamics = "nh";
  }
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    whichDynamics = whichDynamics + argv[10] + "/";
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    whichDynamics = whichDynamics + "-wca/";
  } else {
    whichDynamics = "langevin/";
    cout << "Setting default harmonic potential" << endl;
    sp.setWallType(simControlStruct::wallEnum::harmonic);
  }
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
    sp.setWallEnergyScale(ew);
  } else if(fixedSides == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
    sp.setWallEnergyScale(ew);
  }else {
    cout << "Setting default rectangular geometry" << endl;
  }
  if(reflect == true) {
    sp.setWallType(simControlStruct::wallEnum::reflect);
  }
  if(dampingDir == true) {
    whichDynamics = "damping";
    dirSample = whichDynamics + argv[6] + "/";
  } else {
    dirSample = whichDynamics + "T" + argv[3] + "/";
  }
  ioSPFile ioSP(&sp);
  // set input and output
  if(justRun == true) {
    outDir = inDir + "dynamics/";
    if(std::experimental::filesystem::exists(outDir) == false) {
      std::experimental::filesystem::create_directory(outDir);
    }
    if(readAndSaveSameDir == true) {
      inDir = outDir;
    }
  } else {
    if (readAndSaveSameDir == true) {//keep running the same dynamics
      readState = true;
      inDir = inDir + dirSample;
      outDir = inDir;
      if(runDynamics == true) {
        if(readNH == true) {
          outDir = outDir + "damping" + argv[6] + "/";
        }
        if(logSave == true) {
          inDir = outDir;
          outDir = outDir + "dynamics-log/";
        }
        if(linSave == true) {
          inDir = outDir;
          outDir = outDir + "dynamics/";
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
        if(dampingDir == true) {
          outDir = inDir + "../" + dirSample;
        } else {
          outDir = inDir + "../../" + dirSample;
        }
      } else {
        if(dampingDir == true) {
          if(std::experimental::filesystem::exists(inDir + dirSample) == false) {
            std::experimental::filesystem::create_directory(inDir + dirSample);
          }
        } else {
          if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
            std::experimental::filesystem::create_directory(inDir + whichDynamics);
          }
        }
        outDir = inDir + dirSample;
        if(readNH == true) {
          inDir = outDir;
        }
      }
      std::experimental::filesystem::create_directory(outDir);
    }
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) * forceUnit << endl;
  damping /= timeUnit;
  ioSP.saveLangevinParams(outDir, damping);
  // initialize simulation
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  waveQ = sp.getSoftWaveNumber();
  range *= LJcut * sigma;
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
      ioSP.saveEnergy(step+initialStep, timeStep, numParticles);
      //ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
      if(fixedSides == true) {
        ioSP.saveParticleFixedWallEnergy(step+initialStep, timeStep, numParticles);
      }
      if(step % checkPointFreq == 0) {
        cout << "Langevin: current step: " << step + initialStep;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        cout << " W/N: " << sp.getParticleWork() / numParticles;
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
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
        //ioSP.saveParticleNeighbors(currentDir);
        //ioSP.saveDumpPacking(currentDir, numParticles, nDim, step);
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
