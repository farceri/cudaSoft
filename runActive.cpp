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
  bool readNH = false, activeDir = true, justRun = false;
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // variables
  bool fixedbc = false, fixedSides = false, roundbc = true, reflect = false, reflectnoise = false;
  bool initAngles = false, readState = true, saveFinal = true, logSave = false, linSave = true;//, saveWork = false;
  long numParticles = atol(argv[9]), nDim = atol(argv[10]), maxStep = atof(argv[6]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[7]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = atof(argv[12]), ew = 1e02, LJcut = 4, cutDistance, cutoff = 0.5, sigma, damping, waveQ, width;
  double forceUnit, timeUnit, timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[8]);
  double Tinject = atof(argv[3]), Dr, tp = atof(argv[4]), driving = atof(argv[5]), range = 3;
  std::string outDir, energyFile, currentDir, potType = argv[11], inDir = argv[1], dirSample, whichDynamics = "active";
  //thrust::host_vector<double> boxSize(nDim);
  if(nDim == 3) {
    LJcut = 2.5;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setLangevinType(simControlStruct::langevinEnum::langevin1);
  if(numParticles < 256) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  sp.setParticleType(simControlStruct::particleEnum::active);
  if(readNH == true) {
    whichDynamics = "nh";
  }
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    whichDynamics = whichDynamics + argv[12] + "/";
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    whichDynamics = "active-wca/";
  } else {
    whichDynamics = "active/";
    cout << "Setting default harmonic potential" << endl;
    sp.setBoxType(simControlStruct::boxEnum::harmonic);
  }
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
    sp.setBoxEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundBox);
    sp.setBoxEnergyScale(ew);
  } else if(fixedSides == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
    sp.setBoxEnergyScale(ew);
  } else {
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
  }
  if(reflect == true) {
    sp.setBoxType(simControlStruct::boxEnum::reflect);
  }
  if(activeDir == true) {
    if(reflect == true) {
      whichDynamics = "reflect-tp";
    } else {
      whichDynamics = "tp";
    }
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
          inDir = inDir + "damping" + argv[8] + "/";
          outDir = outDir + "damping" + argv[8] + "/tp" + argv[4] + "-f0" + argv[5] + "/";
        }
        if(logSave == true) {
          inDir =	outDir;
          outDir = outDir + "dynamics-log/";
        }
        if(linSave == true) {
          inDir =	outDir;
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
        if(activeDir == true) {
          outDir = inDir + "../" + dirSample;
        } else {
          outDir = inDir + "../../" + dirSample;
        }
      } else {
        initAngles = true; // initializing from NVT
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
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) << endl;
  cout << "Activity - Peclet: " << driving * tp / (damping * sigma) << " taup: " << tp << " f0: " << driving << endl;
  damping /= timeUnit;
  driving = driving*forceUnit;
  Dr = 1/(tp*timeUnit);
  //if(saveWork == true) {
  //  width = boxSize[0] * atof(argv[12]);
  //  cout << "Measuring work and active work, fluid width: " << width << " centered in Lx / 2" << endl;
  //}
  sp.setSelfPropulsionParams(driving, tp);
  ioSP.saveLangevinParams(outDir, damping);
  // initialize simulation
  //sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
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
  if(sp.getBoxType() == simControlStruct::boxEnum::reflect) {
    cout << "BOX TYPE: reflective" << endl;
  }
  while(step != maxStep) {
    //sp.softParticleActiveLangevinLoop();
    sp.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      //if(saveWork == true) {
      //  ioSP.saveColumnWorkEnergy(step+initialStep, timeStep, numParticles, width);
      //}
      ioSP.saveAlignEnergy(step+initialStep, timeStep, numParticles);
      //ioSP.saveParticleWallEnergy(step+initialStep, timeStep, numParticles, range);
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
        //ioSP.saveDumpPacking(currentDir, numParticles, nDim, step * timeStep);
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
