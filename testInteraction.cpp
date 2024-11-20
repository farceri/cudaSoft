//
// Author: Francesco Arceri
// Date:   10-03-2021
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
using std::cout;

int main(int argc, char **argv) {
  bool read = true, readState = true, saveFinal = false, linSave = true;
  bool doublelj = false, lj = false, wca = false, alltoall = true;
  bool fixedbc = false, roundbc = true, reflect = false, reflectnoise = false;
  long step = 0, numParticles = atol(argv[4]), nDim = 2, maxStep = atof(argv[3]), updateCount = 0, num1 = 1;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  double ec = 1, LJcut = 4, cutoff = 0.2, cutDistance, timeStep = atof(argv[2]), sigma, timeUnit, size;
  double sigma0 = 1, sigma1 = 3, lx = 10, ly = 10, vel1 = -0.1, y0 = 0.2, y1 = 0.7;
  double ea = 1, eb = 1, eab = 1;
  std::string outDir, energyFile, inDir = argv[1], currentDir, dirSample;
  // initialize sp object
  SP2D sp(numParticles, nDim);
  if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedWall);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
  } else {
    cout << "Setting periodic boundary conditins" << endl;
  }
  if(fixedbc == true || roundbc == true) {
    if(reflect == true) {
      sp.setWallType(simControlStruct::wallEnum::reflect);
    } else if(reflectnoise == true) {
      sp.setWallType(simControlStruct::wallEnum::reflectnoise);
    } else {
      cout << "Setting WCA repulsive walls" << endl;
    }
  }
  sp.setEnergyCostant(ec);
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    dirSample = "lj/";
    cout << "Setting Lennard-Jones potential" << endl;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    dirSample = "wca/";
    cout << "Setting WCA potential" << endl;
  } else if(doublelj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::doubleLJ);
    dirSample = "2lj/";
    cout << "Setting double Lennard-Jones potential" << endl;
    sp.setDoubleLJconstants(LJcut, ea, eab, eb, num1);
  } else {
    dirSample = "harmonic/";
    sp.setWallType(simControlStruct::wallEnum::harmonic);
    cout << "Setting Harmonic potential between particles and walls" << endl;
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  ioSPFile ioSP(&sp);
  // set input and output
  if (read == true) {//keep running the same dynamics
    cout << "Read packing" << endl;
    inDir = inDir + dirSample;
    outDir = inDir;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  } else {//start a new dyanmics
    cout << "Initialize new packing" << endl;
    if(numParticles == 3) {
      sp.setThreeParticleTestPacking(sigma0, sigma1, lx, ly, y0, y1, vel1);
      sp.printThreeParticles();
    } else if(numParticles == 2) {
      sp.setTwoParticleTestPacking(sigma0, sigma1, lx, ly, y0, y1, vel1);
      sp.printTwoParticles();
    } else {
      cout << "testInteraction only works for 2 or 3 particles!" << endl;
    }
    if(std::experimental::filesystem::exists(inDir + dirSample) == false) {
      std::experimental::filesystem::create_directory(inDir + dirSample);
    }
    outDir = inDir + dirSample;
  }
  std::experimental::filesystem::create_directory(outDir);
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  timeUnit = sigma0;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma0 << endl;
  cout << "initial velocity on particle 1: " << vel1 << " time step: " << timeStep << endl;
  // initialize simulation
  if(sp.getNeighborType() == simControlStruct::neighborEnum::neighbor) {
    cutDistance = sp.setDisplacementCutoff(cutoff);
    sp.calcParticleNeighbors(cutDistance);
  }
  sp.calcParticleForceEnergy();
  sp.resetUpdateCount();
  sp.setInitialPositions();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  ioSP.saveParticlePacking(outDir);
  ioSP.saveParticleNeighbors(outDir);
  while(step != maxStep) {
    sp.testInteraction(timeStep);
    if(step % saveEnergyFreq == 0) {
      ioSP.saveSimpleEnergy(step, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " Energy: " << sp.getParticleEnergy() / numParticles;
        if(sp.simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
          updateCount = sp.getUpdateCount();
          if(step != 0 && updateCount > 0) {
            cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
          } else {
            cout << " no updates" << endl;
          }
          sp.resetUpdateCount();
        } else {
          cout << endl;
        }
        if(saveFinal == true) {
          ioSP.saveParticlePacking(outDir);
          ioSP.saveParticleNeighbors(outDir);
        }
      }
    }
    //sp.calcParticleNeighborList(cutDistance);
    //sp.checkParticleNeighbors();
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleState(currentDir);
        ioSP.saveParticleNeighbors(currentDir);
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
    ioSP.saveParticleNeighbors(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
