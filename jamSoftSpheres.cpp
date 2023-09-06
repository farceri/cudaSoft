//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/DPM2D.h"
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
  bool read = false, readState = false, increasedStep1 = false, increasedStep2 = false;
  long numParticles = 16384, nDim = 2, numVertexPerParticle = 32; // this is just a default variable to initialize the dpm object
  long iteration = 0, maxIterations = 1e06, minStep = 20, numStep = 0, overJamCount = 0;
  long maxStep = 1e04, step = 0, maxSearchStep = 1500, searchStep = 0, repetition;
  long printIter = int(maxIterations / 10), printFreq = int(maxStep / 10), updateFreq = 1e03;
  double polydispersity = 0.20, previousPhi, currentPhi, deltaPhi = 1e-02, scaleFactor, newTimeStep;
  double cutDistance = 1., forceTollerance0 = 1e-10, pressureTollerance = 1e-08, phi1 = 0.4, phi2 = 0.84;
  double forceCheck, previousForceCheck, energyCheck, energyTollerance = 1e-20, forceTollerance;
  double FIREStep = 2e-03, dt = 5e-03, phi0 = 0.2, phiTh = 0.88, pressure;
  double ec = 240, Tinject = 1e-03, sigma, damping, inertiaOverDamping = 10, timeStep, timeUnit;
  bool jamCheck = 0, overJamCheck = 0, underJamCheck = 0;
  std::string currentDir, outDir = argv[1], inDir;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
  thrust::host_vector<double> particlePos(numParticles * nDim);
  thrust::host_vector<double> particleRad(numParticles);
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[2];
    inDir = outDir + inDir;
    ioDPM.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    particlePos = dpm.getParticlePositions();
    particleRad = dpm.getParticleRadii();
    if(readState == true) {
      ioDPM.readParticleState(inDir, numParticles, nDim);
    }
  } else {
    dpm.setPolyRandomSoftParticles(phi0, polydispersity);
  }
  dpm.setEnergyCosts(0, 0, 0, ec);
  currentPhi = dpm.getParticlePhi();
  cout << "Current phi: " << currentPhi << endl;
  previousPhi = currentPhi;
  while (searchStep < maxSearchStep) {
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
    iteration = 0;
    forceCheck = dpm.getParticleMaxUnbalancedForce();
    energyCheck = dpm.getParticleEnergy();
    repetition = 0;
    previousForceCheck = 0;
    newTimeStep = FIREStep;
    forceTollerance = forceTollerance0 / dpm.getMeanParticleSigma();
    while((forceCheck > forceTollerance) && (iteration != maxIterations)) {
      dpm.particleFIRELoop();
      forceCheck = dpm.getParticleMaxUnbalancedForce();
      energyCheck = dpm.getParticleEnergy() / numParticles;
      if(iteration % printIter == 0 && iteration != 0) {
        cout << "FIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
        cout << " energy: " << energyCheck << "\n" << endl;
        if(previousForceCheck <= forceCheck) {
          repetition += 1;
        }
        if(repetition > 3) {
          newTimeStep /= 2;
          dpm.setTimeStepFIRE(newTimeStep);
          cout << "Dividing the time step in half" << endl;
          repetition = 0;
        }
        previousForceCheck = forceCheck;
      }
      if(iteration % updateFreq == 0) {
        dpm.calcParticleNeighborList(cutDistance);
      }
      iteration += 1;
    }
    pressure = dpm.getParticleVirialPressure();
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
    cout << " energy: " << energyCheck << endl;
    // save minimized configuration
    std::string currentDir = outDir + std::to_string(dpm.getParticlePhi()) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticleConfiguration(currentDir);
    // check configuration after energy minimization
    //jamCheck = (energyCheck < 2.0 * energyTollerance && energyCheck > energyTollerance);
    //overJamCheck = (energyCheck > 2.0 * energyTollerance);
    //underJamCheck = (energyCheck < energyTollerance);
    jamCheck = (forceCheck < 2.0 * forceTollerance && forceCheck > forceTollerance);
    overJamCheck = (forceCheck > 2.0 * forceTollerance);
    underJamCheck = (forceCheck < forceTollerance);
    if(jamCheck) {
      cout << "Compression step: " << searchStep;
      cout << " Found jammed configuration, pressure: " << setprecision(precision) << pressure;
      cout << " packing fraction: " << currentPhi << endl;
    } else {
      // compute scaleFactor with binary search
      if(underJamCheck) {
        scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found underjammed configuration, scaleFactor: " << scaleFactor;
        // save most recent underjammed configuration
        previousPhi = currentPhi;
        particlePos = dpm.getParticlePositions();
        particleRad = dpm.getParticleRadii();
      } else if(overJamCheck) {
        deltaPhi *= 0.5;
        scaleFactor = sqrt((previousPhi + deltaPhi) / previousPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found overjammed configuration, scaleFactor: " << scaleFactor;
        // copy back most recent underjammed configuration and compress half much
        dpm.setParticlePositions(particlePos);
        dpm.setParticleRadii(particleRad);
        if(currentPhi > phi2) {
          overJamCount += 1;
        }
        if(overJamCount > 10) {
          // stop the search after finding 10 overjammed configurations
          searchStep = maxSearchStep;
        }
      }
      if(currentPhi > phi1) {
        // run dynamics
        sigma = dpm.getMeanParticleSigma();
        damping = sqrt(inertiaOverDamping) / sigma;
        timeUnit = 1 / damping;
        timeStep = dpm.setTimeStep(dt * timeUnit);
        cout << "\nRun dynamics, time step: " << timeStep << endl;
        step = 0;
        // thermalize packing after each energy minimization
        dpm.initSoftParticleLangevin(Tinject, damping, true); // readState = false
        while(step != maxStep) {
          dpm.softParticleLangevinLoop();
          if(step % printFreq == 0) {
            cout << "Brownian: current step: " << step;
            cout << " U/N: " << dpm.getParticleEnergy() / numParticles;
            cout << " T: " << dpm.getParticleTemperature() << endl;
          }
          if(step % updateFreq == 0) {
            dpm.calcParticleNeighborList(cutDistance);
          }
          step += 1;
        }
      }
      dpm.scaleParticles(scaleFactor);
      currentPhi = dpm.getParticlePhi();
      cout << "\nNew packing fraction: " << currentPhi << endl;
      searchStep += 1;
      // update parameters as the packing gets closer to jamming
      if(currentPhi > phi1 && increasedStep1 == false) {
        deltaPhi /= 5;
        energyTollerance = 1e-18;
        increasedStep1 = true;
      }
      if(currentPhi > phi2 && increasedStep2 == false) {
        deltaPhi /= 5;
        energyTollerance = 1e-16;
        increasedStep2 = true;
      }
    }
  }

  return 0;
}
