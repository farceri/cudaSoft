//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/SP2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
#include "include/FIRE.h"
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
  bool readState = true, biaxial = true, save = false, saveCurrent, saveForce = false, centered = false;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction = 1;
  double timeStep = atof(argv[2]), forceUnit, timeUnit, LJcut = 4, damping, inertiaOverDamping = atof(argv[9]), width;
  double ec = atof(argv[10]), cutDistance, cutoff = 0.5, sigma,  waveQ, Tinject = atof(argv[3]), range = 3, strainFreq = 0.01;
  double strain, otherStrain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]);
  std::string inDir = argv[1], strainType = argv[11], potType = argv[12], outDir, currentDir, energyFile, dirSample;
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
  SP2D sp(numParticles, nDim);
  if(strainType == "compress") {
    direction = 0;
    if(biaxial == true) {
      dirSample = "nvt-biaxial-comp";
    } else {
      dirSample = "nvt-comp";
    }
  } else if(strainType == "extend") {
    direction = 1;
    if(biaxial == true) {
      dirSample = "nvt-biaxial-ext";
    } else {
      dirSample = "nvt-ext";
    }
  } else {
    cout << "Please specify a strain type between compression and extension" << endl;
    exit(1);
  }
  if(saveForce == true) {
    dirSample += "-wall";
  }
  sp.setEnergyCostant(ec);
  if(potType == "lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    sp.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
  } else if(potType == "harmonic") {
    cout << "Default harmonic potential" << endl;
  } else {
    cout << "Please specify a potential type between lj, wca and harmonic" << endl;
    exit(1);
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + dirSample + argv[5] + "-tmax" + argv[7] + "/";
  //outDir = inDir + dirSample + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + dirSample + argv[5] + "-tmax" + argv[7] + "/strain" + argv[6] + "/";
    //inDir = inDir + dirSample + "/strain" + argv[8] + "/";
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  } else {
    strain = strainStep;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    initBoxSize = sp.getBoxSize();
  }
  double boxRatio = initBoxSize[direction] / initBoxSize[!direction];
  double targetBoxRatio = 1 / boxRatio;
  cout << "Direction: " << direction << " other direction: " << !direction;
  cout << " starting from box ratio: " << boxRatio << " target: " << targetBoxRatio << endl;
  std::experimental::filesystem::create_directory(outDir);
  if(save == false) {
    currentDir = outDir;
    energyFile = outDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    linFreq = checkPointFreq;
  }
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  // save initial configuration
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  forceUnit = ec / sigma;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " force: " << forceUnit << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << " noise magnitude: " << sqrt(2*damping*Tinject) * forceUnit << endl;
  ioSP.saveLangevinParams(outDir, damping);
  range *= LJcut * sigma;
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  // strain by strainStep up to maxStrain
  long countStep = 0;
  long saveFreq = int(strainFreq / strainStep);
  if(saveFreq % 10 != 0) saveFreq += 1;
  cout << "Saving frequency: " << saveFreq << endl;
  boxSize = sp.getBoxSize();
  while (strain < (maxStrain + strainStep) || (boxSize[direction]/boxSize[!direction]) > targetBoxRatio) {
    if(biaxial == true) {
      newBoxSize[direction] = (1 + strain) * initBoxSize[direction];
      otherStrain = -strain / (1 + strain);
      newBoxSize[!direction] = (1 + otherStrain) * initBoxSize[!direction];
      if(direction == 1) {
        cout << "\nStrain y: " << strain << ", x: " << otherStrain << endl;
      } else {
        cout << "\nStrain x: " << strain << ", y: " << otherStrain << endl;
      }
      if(centered == true) {
        sp.applyCenteredBiaxialExtension(newBoxSize, strainStep, direction);
      } else {
        sp.applyBiaxialExtension(newBoxSize, strainStep, direction);
      }
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + strain) * initBoxSize[direction];
      if(centered == true) {
        sp.applyCenteredUniaxialExtension(newBoxSize, strainStep, direction);
      } else {
        sp.applyUniaxialExtension(newBoxSize, strainStep, direction);
      }
      cout << "\nStrain: " << strain << endl;
    }
    boxSize = sp.getBoxSize();
    width = boxSize[0] * 0.5;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1];
    cout << ", box ratio: " << boxSize[direction] / boxSize[!direction] << endl;
    cout << "Abox / Abox0: " << boxSize[0]*boxSize[1]/initBoxSize[0]*initBoxSize[1] << endl;
    saveCurrent = false;
    if((countStep + 1) % saveFreq == 0) {
      cout << "SAVING AT STRAIN: " << strain << endl;
      saveCurrent = true;
      currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      sp.setInitialPositions();
      if(save == true) {
        energyFile = currentDir + "energy.dat";
        ioSP.openEnergyFile(energyFile);
      }
    }
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
    sp.resetUpdateCount();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    while(step != maxStep) {
      sp.softParticleLangevinLoop();
      if((step + 1) % linFreq == 0) {
        if(saveCurrent == true and save == true) {
          if(saveForce == true) {
            if(centered == true) {
              ioSP.saveParticleColumnWallEnergy(step, timeStep, numParticles, range, width);
            }
            ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleEnergy(step, timeStep, numParticles);
          }
        } else {
          if(saveForce == true) {
            if(centered == true) {
              ioSP.saveParticleColumnWallEnergy(step + countStep * maxStep, timeStep, numParticles, range, width);
            }
            ioSP.saveParticleWallEnergy(step + countStep * maxStep, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleEnergy(step + countStep * maxStep, timeStep, numParticles);
          }
        }
      }
      if((step + 1) % checkPointFreq == 0) {
        if(saveCurrent == true) {
          ioSP.saveParticlePacking(currentDir);
        }
      }
      step += 1;
    }
    cout << "NVT: current step: " << step;
    cout << " E/N: " << sp.getParticleEnergy() / numParticles;
    cout << " W/N: " << sp.getParticleWork() / numParticles;
    cout << " T: " << sp.getParticleTemperature();
    cout << " ISF: " << sp.getParticleISF(waveQ);
    updateCount = sp.getUpdateCount();
    if(updateCount > 0) {
      cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
    } else {
      cout << " no updates" << endl;
    }
    countStep += 1;
    // save current configuration
    if(saveCurrent == true) {
      ioSP.saveParticlePacking(currentDir);
      if(save == true) {
        ioSP.closeEnergyFile();
      }
    }
    strain += strainStep;
  }
  if(save == false) {
    ioSP.closeEnergyFile();
  }
  return 0;
}
