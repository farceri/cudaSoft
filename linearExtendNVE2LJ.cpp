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
  bool readState = true, biaxial = true, reverse = false, exponential = false, equilibrate = false, saveFinal = true;
  bool adjustWall = true, adjustGlobal = false, save = false, saveCurrent, saveForce = false, saveStress = false;
  long step, maxStep = atof(argv[7]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 2);
  long numParticles = atol(argv[8]), nDim = 2, updateCount = 0, direction = 1, num1 = atol(argv[9]), initMaxStep = 1e07;
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, strain, otherStrain, strainFreq = 0.01;
  double ec = 1, cutDistance, cutoff = 0.5, sigma, waveQ, Tinject = atof(argv[3]), range = 3, prevEnergy = 0;
  double ea = atof(argv[10]), eb = ea, eab = 0.5, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[6]);
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
  if (exponential == true) {
    dirSample += "-exp";
  }
  if (saveForce == true) {
    dirSample += "-wall";
  }
  if(saveStress == true) {
    dirSample += "-stress";
  }
  if (adjustWall == true) {
    dirSample += "-adjust";
    if (adjustGlobal == true) {
      dirSample += "-global";
    }
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
  ioSP.saveParticlePacking(outDir);
  sigma = 2 * sp.getMeanParticleSigma();
  timeUnit = sigma / sqrt(ea);// sqrt(m sigma^2 / epsilon)
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
    if(adjustWall == true) {
      prevEnergy = sp.getParticleEnergy();
      previousEnergy = sp.getParticleEnergies();
      cout << "Energy before extension - E/N: " << prevEnergy / numParticles << endl;
    }
    if(biaxial == true) {
      if(exponential == true) {
        newBoxSize[direction] = exp(strainStep) * boxSize[direction];
        newBoxSize[!direction] = exp(-strainStep) * boxSize[!direction];
        if(direction == 1) {
          cout << "\nStrain y: " << strain << ", x: " << -strain << endl;
        } else {
          cout << "\nStrain x: " << strain << ", y: " << -strain << endl;
        }
        sp.applyBiaxialExpExtension(newBoxSize, strainStep, direction);
      } else {
        newBoxSize[direction] = (1 + strain) * initBoxSize[direction];
        otherStrain = -strain / (1 + strain);
        newBoxSize[!direction] = (1 + otherStrain) * initBoxSize[!direction];
        if(direction == 1) {
          cout << "\nStrain y: " << strain << ", x: " << otherStrain << endl;
        } else {
          cout << "\nStrain x: " << strain << ", y: " << otherStrain << endl;
        }
        sp.applyCenteredBiaxialExtension(newBoxSize, strainStep, direction);
      }
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
    // adjust kinetic energy to preserve energy conservation
    if(adjustWall == true) {
      cout << "Energy after extension - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustLocalKineticEnergy(previousEnergy, direction);
      cout << "Energy after local adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      if (adjustGlobal == true) {
        sp.adjustKineticEnergy(prevEnergy);
        cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      }
      cout << endl;
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
          } else if(saveStress == true) {
            ioSP.saveParticleStressEnergy(step, timeStep, numParticles, range);
          } else {
            ioSP.saveStrainSimpleEnergy(step, timeStep, numParticles, strain);
          }
        } else {
          if(saveForce == true) {
            ioSP.saveParticleWallEnergy(step + saveStep * maxStep, timeStep, numParticles, range);
          } else if(saveStress == true) {
            ioSP.saveParticleStressEnergy(step + saveStep * maxStep, timeStep, numParticles, range);
          } else {
            ioSP.saveStrainSimpleEnergy(step + saveStep * maxStep, timeStep, numParticles, strain);
          }
        }
      }
      step += 1;
    }
    if(saveCurrent == true) {
      ioSP.saveParticlePacking(currentDir);
    }
    cout << "NVE2LJ: current step: " << step;
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
      }
      else {
        backward = (strain > 0);
      }
    } else {
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
