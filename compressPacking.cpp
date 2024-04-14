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
  bool read = true, readState = true, lj = true, wca = false;
  bool gforce = false, fixedbc = false, alltoall = false, nve = true;
  long numParticles = atol(argv[4]), nDim = 2;
  long iteration = 0, maxIterations = 1e05, minStep = 20, numStep = 0;
  long maxStep = 5e04, step = 0, maxSearchStep = 1500, searchStep = 0, num1;
  long printFreq = int(maxStep / 10), updateCount = 0, saveEnergyFreq = int(printFreq / 10);
  double polydispersity = 0.2, previousPhi, currentPhi, deltaPhi = 1e-02, scaleFactor, prevEnergy = 0;
  double mass = 1, LJcut = 4, forceTollerance = 1e-08, waveQ, FIREStep = 1e-02, dt = atof(argv[2]);
  double ec = 1, ew = 1e02, Tinject = atof(argv[3]), damping, inertiaOverDamping = 10, phi0 = 0.06, phiTh = 0.8;
  double cutDistance, cutoff = 1.5, timeStep, timeUnit, sigma, lx = atof(argv[5]), ly = atof(argv[6]), gravity = 9.8e-04;
  std::string currentDir, outDir = argv[1], inDir, energyFile;
  thrust::host_vector<double> boxSize(nDim);
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setEnergyCostant(ec);
  if(gforce == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
  } else if(fixedbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedBox);
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[7];
    inDir = outDir + inDir + "/";
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    sigma = 2 * sp.getMeanParticleSigma();
    if(readState == true) {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing
    sp.setScaledPolyRandomParticles(phi0, polydispersity, lx, ly);
    //sp.setScaledMonoRandomParticles(phi0, lx, ly);
    sp.scaleParticlePacking();
    sigma = 2 * sp.getMeanParticleSigma();
    sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    sp.setParticleMassFIRE();
    cutDistance = sp.setDisplacementCutoff(cutoff);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    sp.resetUpdateCount();
    sp.setInitialPositions();
    cout << "Generate initial configurations with FIRE \n" << endl;
    while((sp.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      sp.particleFIRELoop();
      if(iteration % printFreq == 0 && iteration != 0) {
        cout << "FIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
        cout << " energy: " << sp.getParticleEnergy() << endl;
        updateCount = sp.getUpdateCount();
        sp.resetUpdateCount();
      }
      iteration += 1;
    }
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << sp.getParticleEnergy() << "\n" << endl;
  }
  if(lj == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    cout << "Setting Lennard-Jones potential" << endl;
    sp.setLJcutoff(LJcut);
  } else if(wca == true) {
    sp.setPotentialType(simControlStruct::potentialEnum::WCA);
    cout << "Setting WCA potential" << endl;
  } else {
    cout << "Setting Harmonic potential" << endl;
  }
  if(gforce == true) {
    sp.setGravityType(simControlStruct::gravityEnum::on);
    sp.setGravity(gravity, ew);
  }
  // quasistatic thermal compression
  currentPhi = sp.getParticlePhi();
  cout << "current phi: " << currentPhi << ", average size: " << sigma << endl;
  previousPhi = currentPhi;
  timeUnit = sigma / sqrt(ec);
  timeStep = sp.setTimeStep(dt*timeUnit);
  if(nve == true) {
    cout << "Time step: " << timeStep << ", Tinject: " << Tinject << endl;
  } else {
    damping = sqrt(inertiaOverDamping) / sigma;
    cout << "Time step: " << timeStep << ", damping: " << damping << endl;
  }
  if(nve == true) {
    sp.initSoftParticleNVE(Tinject, readState);
  } else {
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  }
  // initilize velocities only the first time
  while (searchStep < maxSearchStep) {
    currentDir = outDir + std::to_string(sp.getParticlePhi()).substr(0,4) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    cutDistance = sp.setDisplacementCutoff(cutoff);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    sp.resetUpdateCount();
    sp.setInitialPositions();
    waveQ = sp.getSoftWaveNumber();
    // equilibrate dynamics
    step = 0;
    // remove energy injected by compression
    if(nve == true && searchStep != 0) {
      cout << "Energy after compression - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustKineticEnergy(prevEnergy);
      cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
    }
    while(step != maxStep) {
      if(nve == true) {
        sp.softParticleNVELoop();
      } else {
        sp.softParticleLangevinLoop();
      }
      if(step % saveEnergyFreq == 0) {
        ioSP.saveParticleSimpleEnergy(step, timeStep, numParticles);
      }
      if(step % printFreq == 0) {
        cout << "Compression: current step: " << step;
        cout << " E/N: " << sp.getParticleEnergy() / numParticles;
        cout << " T: " << sp.getParticleTemperature();
        cout << " ISF: " << sp.getParticleISF(waveQ);
        if(sp.simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
          updateCount = sp.getUpdateCount();
          if(step != 0 && updateCount > 0) {
            cout << " number of updates: " << updateCount << " frequency " << printFreq / updateCount << endl;
          } else {
            cout << " no updates" << endl;
          }
          sp.resetUpdateCount();
        } else {
          cout << endl;
        }
      }
      //sp.calcParticleNeighborList(cutDistance);
      //sp.checkParticleNeighbors();
      step += 1;
    }
    prevEnergy = sp.getParticleEnergy();
    cout << "Energy before compression - E/N: " << prevEnergy / numParticles << endl;
    // save minimized configuration
    ioSP.saveParticlePacking(currentDir);
    ioSP.saveParticleNeighbors(currentDir);
    ioSP.closeEnergyFile();
    //ioSP.saveDumpPacking(currentDir, numParticles, nDim, 0);
    // check if target density is met
    if(currentPhi > phiTh) {
      cout << "\nTarget density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      if(nDim == 2) {
        scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      } else if(nDim == 3) {
        scaleFactor = cbrt((currentPhi + deltaPhi) / currentPhi);
      } else {
        cout << "ScaleFactor: only dimensions 2 and 3 are allowed!" << endl;
      }
      sp.scaleParticles(scaleFactor);
      sp.scaleParticlePacking();
      boxSize = sp.getBoxSize();
      currentPhi = sp.getParticlePhi();
      if(nDim == 2) {
        cout << "\nNew phi: " << currentPhi << " Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
      } else if(nDim == 3) {
        cout << "\nNew phi: " << currentPhi << " Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " Lz: " << boxSize[2] << " scale: " << scaleFactor << endl;
      } else {
        cout << "BoxSize: only dimensions 2 and 3 are allowed!" << endl;
      }
      searchStep += 1;
    }
  }
  return 0;
}
