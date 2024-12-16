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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  bool readState = false, initAngles = false, initWall = false, saveFinal = true, logSave = false, linSave = true;
  // variables
  long maxStep = atof(argv[5]), initialStep = atof(argv[6]), numParticles = atol(argv[7]), nDim = 2;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 1, timeStep = atof(argv[2]), timeUnit, forceUnit, alphaUnit, sigma, LJcut = 4, cutDistance, cutoff = 0.5, waveQ;
  double ew = 10*ec, Tinject = 2, Rvicsek = 1.5, Jvicsek = atof(argv[3]), driving = 2, damping = 1, tp = atof(argv[4]);
  double ea = 1e02*ec, el = 1e03*ec, eb = 1e02*ec;
  std::string outDir, currentDir, dirSample, energyFile, wallDir, whichDynamics = "active/";
  std::string inDir = argv[1], potType = argv[8], particleType = argv[9], wallType = argv[10], dynType = argv[11];
  // initialize sp object
	SP2D sp(numParticles, nDim);
  if(particleType == "vicsek") {
    whichDynamics = "vicsek/";
    sp.setParticleType(simControlStruct::particleEnum::vicsek);
  } else if(particleType == "active") {
    sp.setParticleType(simControlStruct::particleEnum::active);
  } else {
    cout << "Particles are set to be passive!" << endl;
  }
  sp.setEnergyCostant(ec);
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
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  if(wallType == "rigid") {
    whichDynamics = whichDynamics + "rigid/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::rigid);
  } else if(wallType == "mobile") {
    whichDynamics = whichDynamics + "mobile/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::mobile);
    sp.setWallShapeEnergyScales(ea, el, eb);
  } else if(wallType == "plastic") {
    whichDynamics = whichDynamics + "plastic/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::plastic);
    sp.setWallShapeEnergyScales(ea, el, eb);
  }else if(wallType == "reflect") {
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
  if(dynType == "langevin") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
    damping = atof(argv[12]);
    whichDynamics = whichDynamics + dynType + argv[12] + "/";
  } else {
    sp.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
    readState = true;
    cout << "Setting default overdamped brownian dynamics" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }
  if(particleType == "vicsek") dirSample = whichDynamics + "j" + argv[3] + "-tp" + argv[4] + "/";
  else dirSample = whichDynamics + "tp" + argv[4] + "/";
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
      initWall = true;
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) ioSP.readParticleState(inDir, numParticles, nDim, initAngles, initWall);
  if(initAngles == true) sp.initializeParticleAngles();
  if(initWall == true) sp.initializeWall();
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sigma = sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  alphaUnit = ec / (sigma * sigma);
  Jvicsek = Jvicsek / (PI * Rvicsek * Rvicsek);
  if(dynType == "langevin") driving = 2. * damping;
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Noise - damping: " << damping << " driving: " << driving << " taup: " << tp << " magnitude: " << sqrt(2 * timeStep / tp) << endl;
  if(particleType == "vicsek") {
    cout << "Vicsek - radius: " << Rvicsek << " strength: " << Jvicsek << " magnitude: " << Jvicsek * timeStep << endl;
  }
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  tp *= timeUnit;
  driving *= forceUnit;
  damping /= timeUnit;
  if(particleType == "vicsek") {
    Jvicsek *= alphaUnit;
    Rvicsek *= sigma;
    sp.setVicsekParams(driving, tp, Jvicsek, Rvicsek);
  } else sp.setSelfPropulsionParams(driving, tp);
  if(wallType == "plastic") sp.setPlasticVariables(damping);
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
    sp.softParticleLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioSP.saveAlignEnergy(step+initialStep, timeStep, numParticles);
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

  return 0;
}
