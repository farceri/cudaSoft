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
  bool readState = true, biaxial = true, reverse = false, equilibrate = false, saveFinal = true;
  bool adjustEkin = false, adjustGlobal = false, adjustTemp = false, save = false, saveCurrent, saveForce = false;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 2);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction = 1, initMaxStep = 1e07;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, otherStrain, range = 3, prevEnergy = 0.0;
  double ec = atof(argv[9]), cutDistance, cutoff = 0.5, sigma, waveQ, Tinject = atof(argv[3]), strain, strainFreq = 0.01;
  double maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]);
  std::string inDir = argv[1], outDir, currentDir, strainType = argv[10], energyFile, dirSample = "nve-ext", dirSave = "strain";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
  thrust::host_vector<double> previousEnergy(numParticles);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(strainType == "compress") {
    direction = 0;
    if(biaxial == true) {
      dirSample = "nve-biaxial-comp";
    } else {
      dirSample = "nve-comp";
    }
  } else if(strainType == "extend") {
    direction = 1;
    if(biaxial == true) {
      dirSample = "nve-biaxial-ext";
    } else {
      dirSample = "nve-ext";
    }
  } else {
    cout << "Please specify a strain type between compression and extension" << endl;
    exit(1);
  }
  if (reverse == true) {
    dirSave = "front";
    dirSample += "-rev";
  }
  if (equilibrate == true) {
    dirSample += "-eq";
  }
  if(saveForce == true) {
    dirSample += "-wall";
  }
  if(adjustEkin == true) {
    dirSample += "-adjust";
    if(adjustGlobal == true) {
      dirSample += "-global";
    }
  }
  sp.setEnergyCostant(ec);
  sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
  sp.setLJcutoff(LJcut);
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
  }
  if(readState == true) {
    ioSP.readParticleState(inDir, numParticles, nDim);
  }
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma/sqrt(ec);//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " epsilon: " << ec;
  if(readState == false) {
    cout << " Tinject: " << Tinject << endl;
  } else {
    cout << endl;
  }
  range *= LJcut * sigma;
  sp.initSoftParticleNVE(Tinject, readState);
  cutDistance = sp.setDisplacementCutoff(cutoff);
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  if(equilibrate == true) {
    // run NVE at zero strain to make sure the system is in equilibrium
    step = 0;
    while(step != initMaxStep) {
      sp.softParticleNVELoop();
      step += 1;
    }
    cout << "NVE2LJ: initial equilibration";
    cout << " U/N: " << sp.getParticlePotentialEnergy() / numParticles;
    cout << " T: " << sp.getParticleTemperature() << endl;
  }
  if(adjustEkin == true) {
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
    if(adjustEkin == true) {
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
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1];
    cout << ", box ratio: " << boxSize[direction] / boxSize[!direction] << endl;
    cout << "Abox / Abox0: " << boxSize[0]*boxSize[1]/initBoxSize[0]*initBoxSize[1] << endl;
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
    if(adjustEkin == true) {
      cout << "Energy after extension - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustLocalKineticEnergy(previousEnergy);
      cout << "Energy after local adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      if(adjustGlobal == true) {
        sp.adjustKineticEnergy(prevEnergy);
        cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      }
    }
    sp.resetUpdateCount();
    step = 0;
    waveQ = sp.getSoftWaveNumber();
    while(step != maxStep) {
      sp.softParticleNVELoop();
      if((step + 1) % linFreq == 0) {
        if(saveCurrent == true and save == true) {
          if(saveForce == true) {
            ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleStrainEnergy(step, timeStep, numParticles, strain);
          }
        } else {
          if(saveForce == true) {
            ioSP.saveParticleWallEnergy(step + saveStep * maxStep, timeStep, numParticles, range);
          } else {
            ioSP.saveParticleStrainEnergy(step + saveStep * maxStep, timeStep, numParticles, strain);
          }
        }
      }
      if((step + 1) % checkPointFreq == 0) {
        if(adjustTemp == true) {
          sp.adjustTemperature(Tinject);
        }
      }
      step += 1;
    }
    if(saveCurrent == true) {
      ioSP.saveParticlePacking(currentDir);
    }
    cout << "NVE: current step: " << step;
    cout << " E/N: " << sp.getParticleEnergy() / numParticles;
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
