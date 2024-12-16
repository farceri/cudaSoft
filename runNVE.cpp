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
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false, justRun = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readNH = false, alltoall = false, squarebc = false, roundbc = true, scaleVel = false;
  bool readState = true, saveFinal = true, logSave = false, linSave = true;
  long numParticles = atol(argv[6]), nDim = atol(argv[7]), maxStep = atof(argv[4]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = atof(argv[9]), ew = 10*ec, LJcut = 4, cutoff = 0.5, cutDistance, waveQ;
  double ea = 1e04*ec, el = 1e02*ec, eb = 10*ec;
  double timeStep = atof(argv[2]), Tinject = atof(argv[3]), sigma, timeUnit;
  std::string outDir, energyFile, currentDir, dirSample, whichDynamics = "nve";
  std::string inDir = argv[1], potType = argv[8], wallType = argv[10];
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(numParticles < 256) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  if(readNH == true) {
    whichDynamics = "nh";
  }
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    whichDynamics = whichDynamics + argv[9] + "/";
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    whichDynamics = whichDynamics + "/";
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  } else {
    whichDynamics = whichDynamics + "/";
    cout << "Setting default harmonic potential" << endl;
    sp.setWallType(simControlStruct::wallEnum::harmonic);
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  if(squarebc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
    sp.setWallEnergyScale(ew);
  }
  if(wallType == "rigid") {
    whichDynamics = whichDynamics + "rigid/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::rigid);
  } else if(wallType == "mobile") {
    whichDynamics = whichDynamics + "mobile/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::mobile);
    sp.setWallShapeEnergyScales(ea, el, eb);
  } else if(wallType == "reflect") {
    whichDynamics = whichDynamics + "reflect/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflect);
  } else if(wallType == "fixed") {
    whichDynamics = whichDynamics + "fixed/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::fixed);
  } else {
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  dirSample = whichDynamics + "T" + argv[3] + "/";
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
          //if(initialStep != 0) {
          inDir = outDir;
          //}
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
  if(readState == true) ioSP.readParticleState(inDir, numParticles, nDim);
  else sp.initializeWall();
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ec);//mass is 1 - sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << endl;
  cout << "Tinject: " << Tinject << " time step: " << timeStep << endl;
  // initialize simulation
  if(scaleVel == true) {
    sp.initSoftParticleNVERescale(Tinject);
  } else {
    sp.initSoftParticleNVE(Tinject, readState);
  }
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
    if(scaleVel == true) {
      sp.softParticleNVERescaleLoop();
    } else {
      sp.softParticleNVELoop();
    }
    if(step % saveEnergyFreq == 0) {
      ioSP.saveSimpleEnergy(step+initialStep, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        if(wallType == "mobile") {
          cout << " E/N: " << sp.getTotalEnergy() / numParticles;
        } else {
          cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        }
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        if(wallType == "mobile") {
          cout << " A: " << sp.getWallArea() << " A0: " << sp.getWallArea0();
        }
        if(sp.simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
          updateCount = sp.getUpdateCount();
          if(step != 0 && updateCount > 0) {
            cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
          } else {
            cout << " no updates" << endl;
          }
          sp.resetUpdateCount();
        } else cout << endl;
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
