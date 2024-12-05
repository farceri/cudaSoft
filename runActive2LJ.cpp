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
  // save in "active" directory for all the previous options: activeDir = true
  // read input and save in "dynamics" directory: justRun = true
  bool readNH = true, activeDir = true, justRun = false, conserve = false;
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = false;
  // variables
  bool initAngles = false, readState = true, saveFinal = true, logSave = false, linSave = false;
  long numParticles = atol(argv[9]), nDim = atol(argv[10]), maxStep = atof(argv[6]), num1 = atol(argv[11]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atol(argv[7]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double forceUnit, timeUnit, timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[8]), waveQ;
  double Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]);
  double ec = 1, ea = atof(argv[13]), eb = ea, eab = 0.5, LJcut = 4, cutDistance, cutoff = 0.5, sigma, damping;
  std::string outDir, potType = argv[12], energyFile, currentDir, inDir = argv[1];
  std::string dynType = argv[14], dirSample, whichDynamics = "active";
  if(nDim == 3) {
    LJcut = 2.5;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setParticleType(simControlStruct::particleEnum::active);
  if(dynType == "langevin1") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
  } else if(dynType == "lang1con") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
    cout << "Conserve momentum is true" << endl;
    conserve = true;
  } else if(dynType == "lang2con") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin2);
    cout << "Conserve momentum is true" << endl;
    conserve = true;
  } else {
    dynType = "langevin2";
    sp.setNoiseType(simControlStruct::noiseEnum::langevin2);
  }
  if(readNH == true) {
    whichDynamics = "nh";
  }
  if(potType == "ljwca") {
    whichDynamics = "active-ljwca/";
    sp.setPotentialType(simControlStruct::potentialEnum::LJWCA);
    sp.setEnergyCostant(ec);
    sp.setLJWCAparams(LJcut, num1);
  } else if(potType == "ljmp") {
    whichDynamics = "active-ljmp/";
    sp.setPotentialType(simControlStruct::potentialEnum::LJMinusPlus);
    sp.setEnergyCostant(ec);
    sp.setLJMinusPlusParams(LJcut, num1);
  } else if(potType == "2lj") {
    whichDynamics = whichDynamics + argv[13] + "/";
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
  } else {
    cout << "Please specify a potential type between ljwca, ljmp and 2lj" << endl;
    exit(1);
  }
  if(activeDir == true) {
    readNH = false;
    whichDynamics = "tp";
    dirSample = whichDynamics + argv[4] + "-f0" + argv[5] + "/";
  } else {
    if(readNH == true) {
      dirSample = whichDynamics + "T" + argv[3] + "/";
    } else {
      dirSample = whichDynamics + "tp" + argv[4] + "-f0" + argv[5] + "/";
    }
  }
  ioSPFile ioSP(&sp);
  // set input and output
  if(justRun == true) {
    outDir = inDir + dynType + "/";
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
          inDir = inDir + "damping" + argv[8] + "/";
          outDir = outDir + "damping" + argv[8] + "/tp" + argv[4] + "-f0" + argv[5] + "/";
        }
        inDir =	outDir;
        if(logSave == true) {
          outDir = outDir + dynType + "-log/";
        } else {
          outDir = outDir + dynType + "/";
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
        if(activeDir == true) {
          outDir = inDir + "../" + dirSample;
        } else {
          outDir = inDir + "../../" + dirSample;
        }
      } else {
        initAngles = true;
        if(activeDir == true) {
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
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim, initAngles);
  }
  if(initAngles == true) sp.initializeParticleAngles();
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ea);
  forceUnit = ea / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  cout << "Activity - Peclet: " << driving * tp / (damping * sigma) << " taup: " << tp << " f0: " << driving << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  tp *= timeUnit;
  sp.setSelfPropulsionParams(driving, tp);
  ioSP.saveLangevinParams(outDir, damping);
  // initialize simulation
  sp.initSoftParticleLangevin(Tinject, damping, readState);
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
    sp.softParticleLangevinLoop(conserve);
    if(step % saveEnergyFreq == 0) {
      ioSP.saveEnergyAB(step+initialStep, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Active: current step: " << step + initialStep;
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
