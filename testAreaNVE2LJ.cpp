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
  bool readState = true, save = true, compress = true, biaxial = true, centered = false, rescaleVel = false;
  long step, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 100);
  long numParticles = atol(argv[7]), nDim = 2, minStep = 20, numStep = 0, updateCount = 0, direction = 0, num1 = atol(argv[9]);
  double timeStep = atof(argv[2]), timeUnit, LJcut = 4, strainx, strainStepx;
  double ec = 1, cutDistance, cutoff = 0.5, sigma, waveQ, Tinject = atof(argv[3]), sign = 1, range = 3, prevEnergy = 0;
  double ea = 1, eab = 0.25, eb = 1, strain, maxStrain = atof(argv[4]), strainStep = atof(argv[5]), initStrain = atof(argv[8]);
  std::string inDir = argv[1], outDir, currentDir, timeDir, energyFile, dirSample = "extend";
  thrust::host_vector<double> boxSize(nDim);
  thrust::host_vector<double> initBoxSize(nDim);
  thrust::host_vector<double> newBoxSize(nDim);
	// initialize sp object
	SP2D sp(numParticles, nDim);
  if(compress == true) {
    sign = -1;
    if(biaxial == true) {
      dirSample = "biaxial-comp";
    } else {
      dirSample = "compress";
    }
  } else if(biaxial == true) {
    dirSample = "biaxial-extend";
  }
  if(centered == true) {
    dirSample = dirSample + "-centered";
  }
  sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
  cout << "Setting double Lennard-Jones potential" << endl;
  sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
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
  std::experimental::filesystem::create_directory(outDir);
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
  sp.calcParticleNeighbors(cutDistance);
  sp.calcParticleForceEnergy();
  waveQ = sp.getSoftWaveNumber();
  // strain by strainStep up to maxStrain
  strainStepx = -sign * strainStep / (1 + sign * strainStep);
  while (strain < (maxStrain + strainStep)) {
    prevEnergy = sp.getParticleEnergy();
    cout << "Energy before extension - E/N: " << prevEnergy / numParticles << endl;
    if(biaxial == true) {
      newBoxSize[1] = (1 + sign * strain) * initBoxSize[1];
      strainx = -sign * strain / (1 + sign * strain);
      newBoxSize[0] = (1 + strainx) * initBoxSize[0];
      cout << "strainx: " << strainx << endl;
      if(centered == true) {
        sp.applyCenteredBiaxialExtension(newBoxSize, sign * strainStep, strainStepx);
      } else {
        sp.applyBiaxialExtension(newBoxSize, sign * strainStep, strainStepx);
      }
    } else {
      newBoxSize = initBoxSize;
      newBoxSize[direction] = (1 + sign * strain) * initBoxSize[direction];
      if(centered == true) {
        sp.applyCenteredUniaxialExtension(newBoxSize, sign * strainStep, direction);
      } else {
        sp.applyUniaxialExtension(newBoxSize, sign * strainStep, direction);
      }
    }
    boxSize = sp.getBoxSize();
    cout << "strain: " << sign * strain << ", density: " << sp.getParticlePhi() << endl;
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << ", Abox: " << boxSize[0]*boxSize[1] << endl;
    cout << "old box - Lx0: " << initBoxSize[0] << ", Ly0: " << initBoxSize[1] << ", Abox0: " << initBoxSize[0]*initBoxSize[1] << endl;
    currentDir = outDir + "strain" + std::to_string(strain).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
    // adjust kinetic energy to preserve energy conservation
    cout << "Energy after extension - E/N: " << sp.getParticleEnergy() / numParticles << endl;
    sp.adjustKineticEnergy(prevEnergy);
    sp.calcParticleForceEnergy();
    cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
    sp.resetUpdateCount();
    step = 0;
    sp.setInitialPositions();
    while(step != maxStep) {
      if(step % linFreq == 0) {
        ioSP.saveParticleWallEnergy(step, timeStep, numParticles, range);
        //ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
      }
      sp.softParticleNVELoop();
      if(step % checkPointFreq == 0) {
        cout << "Extend NVE2LJ: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        sp.resetUpdateCount();
        if(save == true) {
          ioSP.saveParticlePacking(currentDir);
        }
      }
      step += 1;
    }
    // save minimized configuration
    if(save == true) {
      ioSP.saveParticlePacking(currentDir);
    }
    strain += strainStep;
    ioSP.closeEnergyFile();
  }
  return 0;
}
