//
// Author: Francesco Arceri
// Date:   11-18-2024
//
// Include C++ header files

#include "include/SP2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
#include "include/defs.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
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
  bool justRun = true, readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  bool readState = true, saveFinal = true, logSave = false, linSave = true;
  bool initAngles = false, initWall = false, scaleRadial = false;
  // input variables
  double timeStep = atof(argv[2]), Jvicsek = atof(argv[3]), tp = atof(argv[4]);
  double damping = atof(argv[5]), scale = atof(argv[12]), roughness = atof(argv[13]);
  long maxStep = atof(argv[6]), initialStep = atof(argv[7]), numParticles = atol(argv[8]), nDim = 2;
  std::string inDir = argv[1], potType = argv[9], wallType = argv[10], alignType = argv[11];
  // step variables
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  // force and noise variables
  double ec = 1, timeUnit, forceUnit, alphaUnit, sigma, cutDistance, cutoff = 0.5;
  double ew = 10*ec, LJcut = 4, waveQ, Tinject = 2, driving = 2, Rvicsek = 1.5;
  double ea = 1e02*ec, el = 1e03*ec, eb = 1e02*ec;
  std::string outDir, currentDir, dirSample, energyFile, wallFile, wallDyn, wallDir, whichDynamics = "active/";

  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setEnergyCostant(ec);
  sp.setParticleType(simControlStruct::particleEnum::vicsek);

  // set alignment type
  if(alignType == "nonAdd") {
    sp.setAlignType(simControlStruct::alignEnum::nonAdditive);
    whichDynamics = "vicsek-na/";
  } else if(alignType == "vel") {
    sp.setAlignType(simControlStruct::alignEnum::velAlign);
    whichDynamics = "/";
  } else {
    sp.setAlignType(simControlStruct::alignEnum::additive);
    if(alignType == "force") whichDynamics = "vicsek-force/";
  }

  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  } else if(potType == "none") {
    sp.setPotentialType(simControlStruct::potentialEnum::none);
    whichDynamics = whichDynamics + "points/";
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(justRun == false) {
    if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
      std::experimental::filesystem::create_directory(inDir + whichDynamics);
    }
  }
  sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
  sp.setWallEnergyScale(ew);
  if(wallType == "reflect") {
    whichDynamics = whichDynamics + "reflect/";
    wallDyn = "reflect";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflect);
  } else if(wallType == "fixed") {
    whichDynamics = whichDynamics + "fixed/";
    wallDyn = "fixed";
    sp.setBoundaryType(simControlStruct::boundaryEnum::fixed);
  } else if(wallType == "rough") {
    //whichDynamics = whichDynamics + "rough" + argv[13] + "/";
    whichDynamics = whichDynamics + "rough/";
    wallDyn = "rough";
    //wallDyn = wallDyn + argv[13];
    sp.setBoundaryType(simControlStruct::boundaryEnum::rough);
  } else if(wallType == "rigid") {
    whichDynamics = whichDynamics + "rigid" + argv[13] + "/";
    wallDyn = "rigid";
    wallDyn = wallDyn + argv[13];
    sp.setBoundaryType(simControlStruct::boundaryEnum::rigid);
  } else if(wallType == "mobile") {
    wallDyn = "mobile";
    whichDynamics = whichDynamics + "mobile/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::mobile);
    sp.setWallShapeEnergyScales(ea, el, eb);
  } else if(wallType == "plastic") {
    whichDynamics = whichDynamics + "plastic/";
    wallDyn = "plastic";
    sp.setBoundaryType(simControlStruct::boundaryEnum::plastic);
    sp.setWallShapeEnergyScales(ea, el, eb);
  } else {
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  sp.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
  cout << "Setting default overdamped brownian dynamics" << endl;
  // set input and output
  ioSPFile ioSP(&sp);
  if (justRun == true) {
    outDir = inDir + wallDyn + "/"; // "long/";
    if(std::experimental::filesystem::exists(outDir) == false) {
      std::experimental::filesystem::create_directory(outDir);
    }
    if (readAndSaveSameDir == false) {
      if(wallType == "rough" || wallType == "rigid" || wallType == "mobile" || wallType == "plastic") {
        initWall = true; // initializing from NVT
        scaleRadial = true; // scale radial coordinates
      }
    } else {
      readState = true;
      inDir = inDir + wallDyn + "/"; // "long/";
      if (runDynamics == true) {
        outDir = outDir + "dynamics";
        if(logSave == true) outDir = outDir + "-log/";
        else outDir = outDir + "/";
        if(std::experimental::filesystem::exists(outDir) == true) {
          inDir = outDir;
        } else {
          std::experimental::filesystem::create_directory(outDir);
        }
      } else {
        outDir = inDir;
      }
    }
  } else {
    whichDynamics = whichDynamics + "damping" + argv[5] + "/";
    dirSample = whichDynamics + "j" + argv[3] + "-tp" + argv[4] + "/";
    if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
      std::experimental::filesystem::create_directory(inDir + whichDynamics);
    }
    if (readAndSaveSameDir == true) {//keep running the same dynamics
      readState = true;
      inDir = inDir + dirSample;
      outDir = inDir;
      if(runDynamics == true) {
        outDir = outDir + "dynamics-vel";
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
        initWall = true;
        if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
          std::experimental::filesystem::create_directory(inDir + whichDynamics);
        }
        outDir = inDir + dirSample;
      }
      std::experimental::filesystem::create_directory(outDir);
    }
  }
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) ioSP.readParticleState(inDir, numParticles, nDim, initAngles, initWall);
  if(initAngles == true) sp.initializeParticleAngles();
  if(initWall == true) sp.initializeWall(roughness);
  if(scaleRadial == true) sp.shrinkRadialCoordinates(scale);
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  if(wallType == "rigid") {
    // output wall file
    wallFile = outDir + "wallDynamics.dat";
    ioSP.openWallFile(wallFile);
  }

  // initialization
  sigma = sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ec);
  alphaUnit = ec / (sigma * sigma);
  forceUnit = ec / sigma;
  Jvicsek = Jvicsek / (PI * Rvicsek * Rvicsek);
  driving = 2. * damping; // this is necessary to keep the temperature equal to input value, es. 2
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Noise - damping: " << damping << " driving: " << driving << " taup: " << tp << " magnitude: " << sqrt(2 * timeStep / tp) << endl;
  cout << "Vicsek - range: " << Rvicsek << " strength: " << Jvicsek << " magnitude: " << Jvicsek * timeStep << endl;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  tp *= timeUnit;
  driving *= forceUnit;
  damping /= timeUnit;
  Jvicsek *= alphaUnit;
  Rvicsek *= sigma;
  sp.setVicsekParams(driving, tp, Jvicsek, Rvicsek);
  ioSP.saveLangevinParams(outDir, damping);

  // initialize integration scheme
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
    sp.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveAlignEnergy(step+initialStep, timeStep, numParticles);
      if(wallType == "rigid") {
        ioSP.saveWallDynamics(step+initialStep, timeStep);
      }
      if(step % checkPointFreq == 0) {
        cout << "Vicsek: current step: " << step + initialStep;
        cout << " E/N: " << sp.getTotalEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        if(wallType == "mobile" || wallType == "plastic") {
          cout << " Twall: " << sp.getWallTemperature();
          cout << " deltaA/A: " << sp.getWallAreaDeviation();
        } else if(wallType == "rigid") cout << " Kwall: " << sp.getWallRotationalKineticEnergy();
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
  if(wallType == "rigid") ioSP.closeWallFile();

  return 0;
}
