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
  bool readNH = false, dampingDir = false, justRun = false;
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // variables
  bool squarebc = false, roundbc = true, fixedSides = false, conserve = false;
  bool readState = true, saveFinal = true, logSave = false, linSave = true;
  long numParticles = atol(argv[7]), nDim = atol(argv[8]), maxStep = atof(argv[4]);
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long initialStep = atof(argv[5]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = atof(argv[10]), ew = 10*ec, LJcut = 4, cutDistance, cutoff = 0.5, waveQ, timeStep = atof(argv[2]);
  double ea = 1e04*ec, el = 1e02*ec, eb = 10*ec;
  double Tinject = atof(argv[3]), damping = atof(argv[6]), sigma, forceUnit, timeUnit;
  std::string inDir = argv[1], potType = argv[9], wallType = argv[11], dynType = argv[12];
  std::string outDir, energyFile, currentDir, dirSample, whichDynamics;
  // initialize sp object
	SP2D sp(numParticles, nDim);
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
  whichDynamics = dynType;
  if(numParticles < 256) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  if(readNH == true) whichDynamics = "nh";
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    whichDynamics = whichDynamics + argv[10] + "/";
    if(nDim == 3) LJcut = 2.5;
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    whichDynamics = whichDynamics + "/";
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
  } else if(fixedSides == true) {
    if(nDim == 3) sp.setGeometryType(simControlStruct::geometryEnum::fixedSides3D);
    else sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
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
  if(dampingDir == true) {
    whichDynamics = "damping";
    dirSample = whichDynamics + argv[6] + "/";
  } else {
    //dirSample = whichDynamics + "T" + argv[3] + "/";
    dirSample = whichDynamics + "damping" + argv[6] + "/";
  }
  ioSPFile ioSP(&sp);
  // set input and output
  if(justRun == true) {
    outDir = inDir + "dynamics/";
    if(std::experimental::filesystem::exists(outDir) == false) {
      std::experimental::filesystem::create_directory(outDir);
    }
    if(readAndSaveSameDir == true) inDir = outDir;
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
        if(std::experimental::filesystem::exists(outDir) == true) inDir = outDir;
        else std::experimental::filesystem::create_directory(outDir);
      }
    } else {//start a new dyanmics
      if(readAndMakeNewDir == true) {
        readState = true;
        if(dampingDir == true) outDir = inDir + "../" + dirSample;
        else outDir = inDir + "../../" + dirSample;
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
        if(readNH == true) inDir = outDir;
      }
      std::experimental::filesystem::create_directory(outDir);
    }
  }
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) ioSP.readParticleState(inDir, numParticles, nDim);
  else sp.initWall();
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
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
      ioSP.saveEnergy(step+initialStep, timeStep, numParticles);
      if(fixedSides == true) {
        ioSP.saveParticleFixedWallEnergy(step+initialStep, timeStep, numParticles);
      }
      if(step % checkPointFreq == 0) {
        cout << "Langevin: current step: " << step + initialStep;
        if(wallType == "mobile") cout << " E/N: " << sp.getTotalEnergy() / numParticles;
        else cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        cout << " W/N: " << sp.getParticleWork() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        if(wallType == "mobile") cout << " A: " << sp.getWallArea() << " A0: " << sp.getWallArea0();
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
