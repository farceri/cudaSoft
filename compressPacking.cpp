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
  bool read = false, readState = false, nve = true, noseHoover = false, scaleVel = false;
  bool squarebc = false, roundbc = false, lj = false, wca = true, gforce = false, alltoall = false;
  long numParticles = atol(argv[4]), nDim = atol(argv[5]);
  long iteration = 0, maxIterations = 1e05, minStep = 20, numStep = 0;
  long maxStep = 1e05, step = 0, maxSearchStep = 1500, searchStep = 0;
  long printFreq = int(maxStep / 10), updateCount = 0, saveEnergyFreq = int(printFreq / 10);
  double polydispersity = 0.2, previousPhi, currentPhi, deltaPhi = 4e-03, scaleFactor, prevEnergy = 0;
  double LJcut = 4, forceTollerance = 1e-08, waveQ, FIREStep = 1e-02, dt = atof(argv[2]), size;
  double ec = 1, ew = 1e02*ec, Tinject = atof(argv[3]), inertiaOverDamping = 10, phi0 = 0.002, phiTh = 0.02;
  double cutDistance, cutoff = 0.5, timeStep, timeUnit, sigma, lx = atof(argv[6]), ly = atof(argv[7]), lz = atof(argv[8]);
  double gravity = 9.8e-04, mass = 10, damping = 1;
  long num1 = int(numParticles / 2);
  if(nDim == 3) {
    LJcut = 2.5;
  }
  std::string currentDir, outDir = argv[1], inDir, energyFile;
  thrust::host_vector<double> boxSize(nDim);
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setEnergyCostant(ec);
  if(squarebc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::squareWall);
    sp.setWallEnergyScale(ew);
  } else if(roundbc == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::roundWall);
    sp.setWallEnergyScale(ew);
  } else if(gforce == true) {
    sp.setGeometryType(simControlStruct::geometryEnum::fixedSides2D);
    sp.setWallEnergyScale(ew);
  } else {
    cout << "Setting default rectangular geometry with periodic boundary conditions" << endl;
  }
  if(alltoall == true) {
    sp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  }
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[9];
    inDir = outDir + inDir + "/";
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    sigma = sp.getMeanParticleSigma();
    if(readState == true) {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  } else {
    // use harmonic potential for FIRE minimization
    sp.setWallType(simControlStruct::wallEnum::harmonic);
    // initialize polydisperse packing
    if(roundbc == true) {
      sp.setRoundScaledPolyRandomParticles(phi0, polydispersity, lx); // lx is box radius for round geometry
    } else {
      sp.setScaledPolyRandomParticles(phi0, polydispersity, lx, ly, lz);
    }
    //sp.setScaledMonoRandomParticles(phi0, lx, ly, lz);
    //sp.setScaledBiRandomParticles(phi0, lx, ly, lz);
    sp.scaleParticlePacking();
    sigma = sp.getMeanParticleSigma();
    sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    sp.setParticleMassFIRE();
    cutDistance = sp.setDisplacementCutoff(cutoff);
    sp.calcParticleNeighbors(cutDistance);
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
    sp.setWallType(simControlStruct::wallEnum::harmonic);
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
  if(nve == true || noseHoover == true) {
    cout << "Time step: " << timeStep << ", Tinject: " << Tinject << endl;
  } else {
    damping = sqrt(inertiaOverDamping) / sigma;
    cout << "Time step: " << timeStep << ", damping: " << damping << endl;
  }
  if(nve == true) {
    sp.initSoftParticleNVE(Tinject, readState);
  } else if(noseHoover == true) {
    sp.initSoftParticleNoseHoover(Tinject, mass, damping, readState);
  } else if(scaleVel == true) {
    sp.initSoftParticleNVE(Tinject, readState); // this is done to inject velocities
    sp.initSoftParticleNVERescale(Tinject);
  } else {
    sp.initSoftParticleLangevin(Tinject, damping, readState);
  }
  currentDir = outDir + "initial/";
  std::experimental::filesystem::create_directory(currentDir);
  ioSP.saveParticlePacking(currentDir);
  // initilize velocities only the first time
  while (searchStep < maxSearchStep) {
    currentDir = outDir + std::to_string(sp.getParticlePhi()).substr(0,5) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioSP.openEnergyFile(energyFile);
    cutDistance = sp.setDisplacementCutoff(cutoff);
    sp.calcParticleNeighbors(cutDistance);
    sp.calcParticleForceEnergy();
    sp.resetUpdateCount();
    sp.setInitialPositions();
    waveQ = sp.getSoftWaveNumber();
    // equilibrate dynamics
    step = 0;
    // remove energy injected by compression
    if(searchStep != 0 && nve == true) {
      sp.calcParticleNeighbors(cutDistance);
      sp.calcParticleForceEnergy();
      cout << "Energy after compression - E/N: " << sp.getParticleEnergy() / numParticles << endl;
      sp.adjustKineticEnergy(prevEnergy);
      sp.calcParticleForceEnergy();
      cout << "Energy after adjustment - E/N: " << sp.getParticleEnergy() / numParticles << endl;
    }
    while(step != maxStep) {
      if(nve == true) {
        sp.softParticleNVELoop();
      } else if(noseHoover == true) {
        sp.softParticleNoseHooverLoop();
      } else if(scaleVel == true) {
        sp.softParticleNVERescaleLoop();
      } else {
        sp.softParticleLangevinLoop();
      }
      if(step % saveEnergyFreq == 0) {
        ioSP.saveSimpleEnergy(step, timeStep, numParticles);
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
      if(nve == true) {
        if(abs(sp.getParticleTemperature() - Tinject) > 1e-02) {
          sp.rescaleParticleVelocity(Tinject);
        }
      }
      step += 1;
    }
    prevEnergy = sp.getParticleEnergy();
    cout << "Energy before compression - E/N: " << prevEnergy / numParticles << endl;
    // save minimized configuration
    ioSP.saveParticlePacking(currentDir);
    //ioSP.saveParticleNeighbors(currentDir);
    //if(noseHoover == true) {
    //  ioSP.saveNoseHooverParams(currentDir);
    //}
    if(nDim == 3) {
      ioSP.saveDumpPacking(currentDir, numParticles, nDim, 0);
    }
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
      currentPhi = sp.getParticlePhi();
      if(roundbc == true) {
        cout << "\nNew phi: " << currentPhi << " box radius: " << sp.getBoxRadius() << " scale: " << scaleFactor << endl;
      } else {
        boxSize = sp.getBoxSize();
        if(nDim == 2) {
          cout << "\nNew phi: " << currentPhi << " Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
        } else if(nDim == 3) {
          cout << "\nNew phi: " << currentPhi << " Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " Lz: " << boxSize[2] << " scale: " << scaleFactor << endl;
        } else {
          cout << "WallSize: only dimensions 2 and 3 are allowed!" << endl;
        }
      }
      searchStep += 1;
    }
  }
  return 0;
}
