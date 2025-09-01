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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  bool readState = false, saveFinal = true, logSave = false, linSave = false;
  bool initAngles = false, squarebc = false, roundbc = true, maxRvicsek = false;
  // input variables
  double timeStep = atof(argv[2]), Jvicsek = atof(argv[3]), tp = atof(argv[4]), damping = atof(argv[5]);
  long maxStep = atof(argv[6]), initialStep = atof(argv[7]), numParticles = atol(argv[8]), nDim = 2;
  std::string inDir = argv[1], potType = argv[9], wallType = argv[10], dynType = argv[11], alignType = argv[12];
  // step variables
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  long step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  // force and noise variables
  double ec = 1, timeUnit, forceUnit, alphaUnit, sigma, cutDistance, cutoff = 0.5; 
  double ew = 10*ec, LJcut = 4, waveQ, Tinject = 2, driving = 2, Rvicsek = 1.5;
  std::string outDir, currentDir, dirSample, energyFile, dampingDir, whichDynamics = "vicsek/";

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
    whichDynamics = "vicsek-vel/";
  } else {
    sp.setAlignType(simControlStruct::alignEnum::additive);
    if(alignType == "force") whichDynamics = "vicsek-force/";
  }

  // set potential type
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
  if(numParticles < 256) sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }

  // set boundary conditions
  if(squarebc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
    sp.setWallEnergyScale(ew);
  }
  if(wallType == "reflect") {
    whichDynamics = whichDynamics + "reflect/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflect);
  } else if(wallType == "noise") {
    whichDynamics = whichDynamics + "noise/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::reflectNoise);
  } else if(wallType == "fixed") {
    whichDynamics = whichDynamics + "fixed/";
    sp.setBoundaryType(simControlStruct::boundaryEnum::fixed);
  } else {
    cout << "Setting default rectangular geometry with periodic boundaries" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }

  // set dynamics type
  if(dynType == "langevin") {
    sp.setNoiseType(simControlStruct::noiseEnum::langevin1);
    Tinject = atof(argv[3]);
    Jvicsek = 1e03;
    tp = 0.;
    whichDynamics = whichDynamics + dynType + argv[5] + "/";
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else if(dynType == "brownian") {
    sp.setNoiseType(simControlStruct::noiseEnum::brownian);
    Tinject = atof(argv[3]);
    Jvicsek = 1e03;
    tp = 0.;
    whichDynamics = whichDynamics + dynType + argv[5] + "/";
    dirSample = whichDynamics + "T" + argv[3] + "/";
  } else {
    sp.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
    whichDynamics = whichDynamics + "damping" + argv[5] + "/";
    if(maxRvicsek == true) whichDynamics = whichDynamics + "maxR/";
    dirSample = whichDynamics + "j" + argv[3] + "-tp" + argv[4] + "/";
    readState = true;
    cout << "Setting default overdamped brownian dynamics" << endl;
  }
  if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
    std::experimental::filesystem::create_directory(inDir + whichDynamics);
  }

  // set input and output
  ioSPFile ioSP(&sp);
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      if(alignType == "vel") {
        outDir = outDir + "dynamics-vel";
      } else {
        outDir = outDir + "dynamics";
      }
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
      initAngles = true;
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
  alphaUnit = ec / (sigma * sigma);
  forceUnit = ec / sigma;
  if(maxRvicsek == true) Rvicsek = sp.getBoxRadius();
  Jvicsek = Jvicsek / (PI * Rvicsek * Rvicsek);
  driving = 2. * damping; // this is necessary to keep the temperature equal to input value, es. 2
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Noise - damping: " << damping << " driving: " << driving << " taup: " << tp << " magnitude: " << sqrt(2 * timeStep / tp) << endl;
  if(dynType == "langevin") cout << "Langevin - T: " << Tinject << " magnitude: " << sqrt(2 * damping * Tinject) << endl;
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
