//
// Author: Francesco Arceri
// Date:   03-22-2024
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
  bool readState = true, biaxial = true, save = false, saveCurrent, saveForce = true, equilibrate = false;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction, num1 = atol(argv[9]), initMaxStep = 1e03;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, damping, inertiaOverDamping = 10, otherStrain;
  double ec = 1, cutDistance, cutoff = 0.5, sigma, waveQ, Tinject = atof(argv[3]), range = 3, strainFreq = 0.02;
  double ea = 2, eb = 2, eab = 0.5, strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]);
  std::string inDir = argv[1], strainType = argv[10], potType = argv[11], outDir, currentDir, timeDir, energyFile, dirSample;
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
  if(potType == "ljwca") {
    sp.setPotentialType(simControlStruct::potentialEnum::LJWCA);
    sp.setEnergyCostant(ec);
    sp.setLJWCAparams(LJcut, num1);
  } else if(potType == "ljmp") {
    sp.setPotentialType(simControlStruct::potentialEnum::LJMinusPlus);
    sp.setEnergyCostant(ec);
    sp.setLJMinusPlusParams(LJcut, num1);
  } else if(potType == "2lj") {
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
  } else {
    cout << "Please specify a potential type between ljwca, ljmp and 2lj" << endl;
    exit(1);
  }
  ioSPFile ioSP(&sp);
  outDir = inDir + dirSample + argv[5] + "-tmax" + argv[6] + "/";
  //outDir = inDir + dirSample + "/";
  if(initStrain != 0) {
    // read initial boxSize
    initBoxSize = ioSP.readBoxSize(inDir, nDim);
    strain = initStrain;
    inDir = inDir + dirSample + argv[5] + "-tmax" + argv[6] + "/strain" + argv[8] + "/";
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
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  ioSP.saveLangevinParams(outDir, sigma, damping);
  range *= LJcut * sigma;
  sp.initSoftParticleLangevin(Tinject, damping, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  if(equilibrate == true) {
    // run NVT at zero strain to make sure the system is in equilibrium
    step = 0;
    while(step != initMaxStep) {
      sp.softParticleLangevinLoop();
      step += 1;
    }
    cout << "NVT2LJ: initial equilibration";
    cout << " U/N: " << sp.getParticlePotentialEnergy() / numParticles;
    cout << " T: " << sp.getParticleTemperature() << endl;
  }
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
      sp.applyBiaxialExtension(newBoxSize, strainStep, direction);
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + strain) * initBoxSize[direction];
      sp.applyUniaxialExtension(newBoxSize, strainStep, direction);
      cout << "\nStrain: " << strain << endl;
    }
    boxSize = sp.getBoxSize();
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
            ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
          }
        } else {
          if(saveForce == true) {
            ioSP.saveParticleWallEnergy(step + countStep * maxStep, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleSimpleEnergy(step + countStep * maxStep, timeStep, numParticles);
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
    cout << "NVT2LJ: current step: " << step;
    cout << " U/N: " << sp.getParticlePotentialEnergy() / numParticles;
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
