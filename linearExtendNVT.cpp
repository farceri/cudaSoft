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
  bool readState = true, biaxial = true, reverse = false, saveFinal = true;
  bool adjustTemp = false, adjustWall = false, adjustGlobal = false, save = false, saveCurrent, saveForce = false;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 2);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction = 1;
  double timeStep = atof(argv[2]), forceUnit, timeUnit, LJcut = 4, damping, inertiaOverDamping = atof(argv[9]), width, tempTh = 1e-03;
  double ec = atof(argv[10]), cutDistance, cutoff = 0.5, sigma,  waveQ, Tinject = atof(argv[3]), range = 3, strainFreq = 0.01;
  double strain, otherStrain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]), prevEnergy = 0.0;
  std::string inDir = argv[1], strainType = argv[11], potType = argv[12], outDir, currentDir, energyFile, dirSample, dirSave = "strain";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
  thrust::host_vector<double> previousEnergy(numParticles);
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
  if (reverse == true) {
    dirSave = "front";
    dirSample += "-rev";
  }
  if(saveForce == true) {
    dirSample += "-wall";
  }
  if (adjustWall == true) {
    dirSample += "-adjust";
    if (adjustGlobal == true)
    {
      dirSample += "-global";
    }
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
    strain = initStrain + strainStep;
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
    if(initStrain != 0) {
      ioSP.reopenEnergyFile(energyFile);
    } else {
      ioSP.openEnergyFile(energyFile);
    }
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
  if (adjustWall == true) {
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
  }
  // strain by strainStep up to maxStrain
  long countStep = 0;
  long saveStep = 0;
  long saveFreq = int(strainFreq / strainStep);
  if(saveFreq % 10 != 0) saveFreq += 1;
  cout << "Saving frequency: " << saveFreq << endl;
  boxSize = sp.getBoxSize();
  //while (strain < (maxStrain + strainStep) || (boxSize[direction]/boxSize[!direction]) > targetBoxRatio) {
  bool switched = false;
  bool forward = (strain < (maxStrain + strainStep));
  bool backward = false;
  while (forward != backward) {
    if (adjustWall == true) {
      prevEnergy = sp.getParticleEnergy();
      previousEnergy = sp.getParticleEnergies();
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
      sp.applyCenteredBiaxialExtension(newBoxSize, strainStep, direction);
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + strain) * initBoxSize[direction];
      sp.applyCenteredUniaxialExtension(newBoxSize, strainStep, direction);
      cout << "\nStrain: " << strain << endl;
    }
    boxSize = sp.getBoxSize();
    width = boxSize[0] * 0.5;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1];
    cout << ", box ratio: " << boxSize[direction] / boxSize[!direction] << endl;
    cout << "Abox / Abox0: " << (boxSize[0]*boxSize[1]) / (initBoxSize[0]*initBoxSize[1]) << endl;
    saveCurrent = false;
    if((countStep + 1) % saveFreq == 0) {
      cout << "SAVING AT STRAIN: " << strain << endl;
      saveCurrent = true;
      currentDir = outDir + dirSave + std::to_string(strain).substr(0,6) + "/";
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
    if(adjustTemp == true) {
      if(abs(Tinject - sp.getParticleTemperature()) > tempTh) {
        sp.adjustTemperature(Tinject);
      }
    }
    if (adjustWall == true) {
      cout << "Energy after extension - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustLocalKineticEnergy(previousEnergy, direction);
      cout << "Energy after local adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      if (adjustGlobal == true)
      {
        sp.adjustKineticEnergy(prevEnergy);
        cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      }
    }
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
            ioSP.saveStrainEnergy(step, timeStep, numParticles, strain);
          }
        } else {
          if(saveForce == true) {
            ioSP.saveParticleWallEnergy(step + saveStep * maxStep, timeStep, numParticles, range);
          } else {
            ioSP.saveStrainEnergy(step + saveStep * maxStep, timeStep, numParticles, strain);
          }
        }
      }
      step += 1;
    }
    if(saveCurrent == true) {
      ioSP.saveParticlePacking(currentDir);
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
    saveStep += 1;
    // save current configuration
    if(saveCurrent == true) {
      ioSP.saveParticlePacking(currentDir);
      if(save == true) {
        ioSP.closeEnergyFile();
      }
    }
    strain += strainStep;
    if (strain < (maxStrain + strainStep)) {
      if (switched == false) {
        forward = (strain < (maxStrain + strainStep));
      } else {
        backward = (strain > 0);
      }
    }
    else {
      if(reverse == false && strain > maxStrain) {
        cout << "strain larger than max " << forward << " " << backward << endl;
        backward = true;
      } else if (reverse == true && switched == false) {
        switched = true;
        strainStep = -strainStep;
        forward = false;
        backward = (strain > 0);
        dirSave = "back";
        countStep = 0;
      }
    }
  }
  if(save == false) {
    ioSP.closeEnergyFile();
  }
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  return 0;
}
