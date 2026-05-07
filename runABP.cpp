//
// Author: Francesco Arceri
// Date:   11-02-2024
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
  bool readState = true, initAngles = false, saveFinal = true, logSave = false, linSave = true;
  // input variables
  double timeStep = atof(argv[2]), tp = atof(argv[3]), driving = atof(argv[4]), damping = atof(argv[5]);
  long maxStep = atof(argv[6]), initialStep = atof(argv[7]), numParticles = atol(argv[8]), nDim = 2;
  std::string inDir = argv[1], potType = argv[9], boxType = argv[10], wallType = argv[11], dynType = argv[12];
  // step variables
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  // force and noise variables
  double ec = 1, ew = 10*ec, LJcut = 4., waveQ, Tinject = 1.;
  double timeUnit, forceUnit, sigma, cutDistance, cutoff = 0.5;
  std::string outDir, currentDir, dirSample, energyFile, whichDynamics = "active/";

  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setEnergyCostant(ec);
  sp.setParticleType(simControlStruct::particleEnum::active);

  // set potential type
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(numParticles < 256) sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }

  // set boundary conditions
  if(boxType == "square") {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  } else if(boxType == "sides2d") {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
    sp.setWallEnergyScale(ew);
  } else if(boxType == "circle") {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
    sp.setWallEnergyScale(ew);
  }
  if(wallType == "reflect") {
    whichDynamics = whichDynamics + "reflect/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflect);
  } else if(wallType == "noise") {
    whichDynamics = whichDynamics + "noise/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflectNoise);
  } else if(wallType == "wall") {
    whichDynamics = whichDynamics + "wall/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::fixed);
  } else {
    whichDynamics = whichDynamics + "pbc/";
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }

  // set dynamics type
  if(dynType == "langevin") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
    whichDynamics = whichDynamics + "langevin" + argv[5] + "/";
  } else {
    sp.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
    whichDynamics = whichDynamics + "damping" + argv[5] + "/";
    readState = true;
    cout << "Setting default driven brownian dynamics" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  dirSample = whichDynamics + "tp" + argv[3] + "-v0" + argv[4] + "/";

  // set input and output
  ioSPFile ioSP(&sp);
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics";
      if(logSave == true) outDir = outDir + "-log/";
      else outDir = outDir + "/";
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
  if(readState == true) ioSP.readParticleState(inDir, numParticles, nDim, initAngles);
  if(initAngles == true) sp.initializeParticleAngles();
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  
  // initialization
  sigma = sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  driving = driving * damping; // get force driving from velocity driving, Fa = v0 * gamma
  if(atof(argv[3]) == 33) tp = sigma * sigma * damping / (3. * Tinject);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Noise - damping: " << damping << " driving: " << driving << " taup: " << tp << " Pe = 3 v_0 tau_p / sigma: " << 3 * (driving / damping) * tp / sigma << endl;
  if(atof(argv[3]) != 33) cout << "Reference rotational time: " << sigma * sigma * damping / (3. * Tinject) << " Pe(D_r = 3 D / sigma^2): " << driving * sigma / Tinject << endl;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  tp *= timeUnit;
  driving *= forceUnit;
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
        cout << "Active: current step: " << step + initialStep;
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
