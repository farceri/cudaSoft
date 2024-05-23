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
  bool readState = true, compress = true, biaxial = true, adjustEkin = false;
  bool save = false, saveCurrent, adjustTemp = false, ljwca = false, ljmp = true;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction = 1, num1 = atol(argv[9]);
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, strain, otherStrain, strainFreq = 0.02;
  double ec = 1, cutDistance, cutoff = 0.5, sigma, waveQ, Tinject = atof(argv[3]), range = 3, prevEnergy = 0;
  double ea = 3, eb = 3, eab = 0.5, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]);
  std::string inDir = argv[1], outDir, currentDir, energyFile, dirSample = "nve-ext";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    direction = 0;
    if(biaxial == true) {
      dirSample = "nve-biaxial-comp";
    } else {
      dirSample = "nve-comp";
    }
  } else if(biaxial == true) {
    dirSample = "nve-biaxial-ext";
  }
  if(adjustEkin == true) {
    dirSample = dirSample + "-adjust";
  }
  if(ljwca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::LJWCA);
    sp.setEnergyCostant(ec);
    sp.setLJWCAparams(LJcut, num1);
  } else if(ljmp == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::LJMinusPlus);
    sp.setEnergyCostant(ec);
    sp.setLJMinusPlusParams(LJcut, num1);
  } else {
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
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
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma;
  if(readState == false) {
    cout << " Tinject: " << Tinject << endl;
  } else {
    cout << endl;
  }
  range *= LJcut * sigma;
  sp.initSoftParticleNVE(Tinject, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  if(adjustEkin == true) {
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
  }
  waveQ = sp.getSoftWaveNumber();
  // strain by strainStep up to maxStrain
  long countStep = 0;
  long saveFreq = int(strainFreq / strainStep);
  if(saveFreq % 10 != 0) saveFreq += 1;
  cout << "Saving frequency: " << saveFreq << endl;
  boxSize = sp.getBoxSize();
  while (strain < (maxStrain + strainStep) or (boxSize[direction]/boxSize[!direction]) > targetBoxRatio) {
    if(adjustEkin == true) {
      prevEnergy = sp.getParticleEnergy();
      cout << "Energy before extension - E/N: " << prevEnergy / numParticles << endl;
    }
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
    // adjust kinetic energy to preserve energy conservation
    if(adjustEkin == true) {
      cout << "Energy after extension - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustKineticEnergy(prevEnergy);
      sp.calcParticleForceEnergy();
      cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
    }
    sp.resetUpdateCount();
    step = 0;
    while(step != maxStep) {
      if((step + 1) % linFreq == 0) {
        if(saveCurrent == true and save == true) {
          //ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
          ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
        } else {
          //ioSP.saveParticleWallEnergy(step + countStep * maxStep, timeStep, numParticles, range);
          ioSP.saveParticleSimpleEnergy(step + countStep * maxStep, timeStep, numParticles);
        }
      }
      sp.softParticleNVELoop();
      if((step + 1) % checkPointFreq == 0) {
        if(adjustTemp == true) {
          sp.adjustTemperature(Tinject);
        }
        if(saveCurrent == true) {
          ioSP.saveParticlePacking(currentDir);
          ioSP.saveParticleEnergies(currentDir);
        }
      }
      step += 1;
    }
    cout << "NVE2LJ: current step: " << step;
    cout << " U/N: " << sp.getParticlePotentialEnergy() / numParticles;
    cout << " T: " << sp.getParticleTemperature();
    cout << " ISF: " << sp.getParticleISF(waveQ);
    updateCount = sp.getUpdateCount();
    cout << " number of updates: " << updateCount << " frequency " << maxStep / updateCount << endl;
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
