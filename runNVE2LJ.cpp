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
#include <tuple>
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
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false, justRun = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readNH = true, alltoall = false, fixedbc = false, scaleVel = false, doubleT = false;
  bool readState = true, saveFinal = true, logSave = false, linSave = false;
  long numParticles = atol(argv[6]), nDim = atol(argv[7]), num1 = atol(argv[8]), updateCount = 0;
  long step, maxStep = atof(argv[4]), initialStep = atol(argv[5]), checkPointFreq = int(maxStep / 10);
  long linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10), firstDecade = 0, multiple = 1, saveFreq = 1;
  double ec = 1, ew = ec, LJcut = 4, cutoff = 0.5, cutDistance, waveQ, timeStep = atof(argv[2]), timeUnit, sigma;
  double ea = atof(argv[11]), eb = ea, eab = 0.5, Tinject = atof(argv[3]), Tinject2 = atof(argv[9]), range = 3;
  std::string outDir, potType = argv[10], energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "nve";
  std::tuple<double, double, double> Temps;
  if(nDim == 3) {
    LJcut = 2.5;
  }
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  }
  if(readNH == true) {
    whichDynamics = "nh";
  }
  if(potType == "2lj") {
    whichDynamics = whichDynamics + argv[11] + "/";
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
    sp.setEnergyCostant(ec);
  } else if(potType == "ljwca") {
    whichDynamics = whichDynamics + "-ljwca/";
    sp.setPotentialType(simControlStruct::potentialEnum::LJWCA);
    sp.setEnergyCostant(ec);
    sp.setLJWCAparams(LJcut, num1);
  } else if(potType == "ljmp") {
    whichDynamics = whichDynamics + "-ljmp/";
    sp.setPotentialType(simControlStruct::potentialEnum::LJMinusPlus);
    sp.setEnergyCostant(ec);
    sp.setLJMinusPlusParams(LJcut, num1);
  } else {
    cout << "Please specify a potential type between ljwca, ljmp and 2lj" << endl;
    exit(1);
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  dirSample = whichDynamics + "T" + argv[3] + "/";
  ioSPFile ioSP(&sp);
  // set input and output
  if(justRun == true) {
    outDir = inDir + "test-dt" + argv[2] + "/";
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
          outDir = outDir + "nve/";
          if(logSave == true) {
            inDir = outDir;
            outDir = outDir + "dynamics-log/";
          }
          if(linSave == true) {
            inDir = outDir;
            outDir = outDir + "dynamics/";
          }
        }
        if(std::experimental::filesystem::exists(outDir) == true) {
          if(initialStep != 0) {
            inDir = outDir;
          }
        } else {
          std::experimental::filesystem::create_directory(outDir);
        }
      }
    } else {//start a new dyanmics
      if(readAndMakeNewDir == true) {
        scaleVel = true;
        readState = true;
        outDir = inDir + "../../" + dirSample;
      } else {
        if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
          std::experimental::filesystem::create_directory(inDir + whichDynamics);
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
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ea);// sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << endl;
  if(scaleVel == true) {
    if(doubleT == true) {
    cout << "T1: " << Tinject << " T2: " << Tinject2 << " time step: " << timeStep << endl;
    } else {
    cout << "Tinject: " << Tinject << " time step: " << timeStep << endl;
    }
  } else {
    cout << "Reading state - time step: " << timeStep << endl;
  }
  // initialize simulation
  if(scaleVel == true) {
    if(doubleT == true) {
      sp.initSoftParticleNVEDoubleRescale(Tinject, Tinject2);
    } else {
      sp.initSoftParticleNVERescale(Tinject);
    }
  } else {
    sp.initSoftParticleNVE(Tinject, readState);
  }
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  sp.setInitialPositions();
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
    if(scaleVel == true) {
      if(doubleT == true) {
        sp.softParticleNVEDoubleRescaleLoop();
      } else {
        sp.softParticleNVERescaleLoop();
      }
    } else {
      sp.softParticleNVELoop();
    }
    if(step % saveEnergyFreq == 0) {
      if(doubleT == true) {
        ioSP.saveParticleDoubleEnergy(step+initialStep, timeStep, numParticles, num1);
      } else {
        //ioSP.saveSimpleEnergy(step+initialStep, timeStep, numParticles);
        ioSP.saveSimplePressureEnergy(step+initialStep, timeStep, numParticles);
      }
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        if(doubleT == true) {
          Temps = sp.getParticleT1T2();
          cout << " T1: " << get<0>(Temps) << " T2: " << get<1>(Temps);
        } else {
          cout << " T: " << sp.getParticleTemperature();
        }
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
          ioSP.saveParticleForces(outDir);
          //ioSP.saveParticleNeighbors(outDir);
          if(nDim == 3) {
            ioSP.saveDumpPacking(outDir, numParticles, nDim, step);
          }
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
    ioSP.saveParticleForces(outDir);
    //ioSP.saveParticleNeighbors(outDir);
    if(nDim == 3) {
      ioSP.saveDumpPacking(outDir, numParticles, nDim, step);
    }
  }
  ioSP.closeEnergyFile();

  return 0;
}
