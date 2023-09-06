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
  bool read = false, readState = false, attraction = false;
  long numParticles = 1024, nDim = 2, numVertexPerParticle = 32; // this is just a default variable to initialize the sp object
  long iteration = 0, maxIterations = 1e05, minStep = 20, numStep = 0;
  long maxStep = 1e04, step = 0, maxSearchStep = 1500, searchStep = 0;
  long printFreq = int(maxStep / 10), updateFreq = 10;
  double polydispersity = 0.2, previousPhi, currentPhi, deltaPhi = 1e-02, scaleFactor, isf = 1;
  double cutDistance = 0.1, forceTollerance = 1e-08, waveQ, FIREStep = 1e-02, dt = atof(argv[2]);
  double ec = 240, Tinject = atof(argv[3]), damping, inertiaOverDamping = 10, phi0 = 0.1, phiTh = 1;
  double timeStep, timeUnit, escale = 1, sigma, l2 = 0.2; // warning: it is running attractive
  std::string currentDir, outDir = argv[1], inDir;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim, numVertexPerParticle);
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[4];
    inDir = outDir + inDir;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    sigma = sp.getMeanParticleSigma();
    sp.setEnergyCosts(0, 0, 0, ec);
    if(attraction == true) {
      sp.setAttractionConstants(escale * sigma, l2); //kc = 1
    }
    if(readState == true) {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing
    sp.setPolyRandomSoftParticles(phi0, polydispersity);
    sigma = sp.getMeanParticleSigma();
    sp.setEnergyCosts(0, 0, 0, ec);
    sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    sp.setParticleMassFIRE();
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    while((sp.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      sp.particleFIRELoop();
      if(iteration % printFreq == 0 && iteration != 0) {
        cout << "\nFIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
        cout << " energy: " << sp.getParticleEnergy() << endl;
      }
      if(iteration % updateFreq == 0) {
        sp.calcParticleNeighborList(cutDistance);
      }
      iteration += 1;
    }
    cout << "\nFIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << sp.getParticleEnergy() << endl;
    ioSP.saveParticlePacking(outDir);
  }
  // quasistatic thermal compression
  currentPhi = sp.getParticlePhi();
  cout << "current phi: " << currentPhi << ", average size: " << sigma << endl;
  if(attraction == true) {
    cout << "attractive constants, l1: " << escale * sigma << " l2: " << l2 << endl;
  }
  previousPhi = currentPhi;
  // initilize velocities only the first time
  while (searchStep < maxSearchStep) {
    damping = sqrt(inertiaOverDamping) / sigma;
    timeUnit = sigma * sigma * damping;//epsilon is 1
    timeStep = sp.setTimeStep(dt * timeUnit);
    cout << "Time step: " << timeStep << ", damping: " << damping << endl;
    sp.initSoftParticleLangevin(Tinject, damping, readState);
    //sp.initSoftParticleNVERA(Tinject, readState);
    //sp.initSoftParticleNVEFixedBoundary(Tinject, readState);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    waveQ = sp.getSoftWaveNumber();
    // equilibrate dynamics
    step = 0;
    isf = 1;
    while(step != maxStep) {
      sp.softParticleLangevinLoop();
      //sp.softParticleNVERALoop();
      //sp.softParticleNVEFixedBoundaryLoop();
      if(step % printFreq == 0) {
        isf = sp.getParticleISF(waveQ);
        cout << "Langevin: current step: " << step;
        cout << " U: " << sp.getParticleEnergy();
        cout << " K: " << sp.getParticleKineticEnergy();
        cout << " ISF: " << isf << endl;
      }
      if(step % updateFreq == 0) {
        sp.calcParticleNeighborList(cutDistance);
      }
      step += 1;
    }
    cout << "Langevin: current step: " << step;
    cout << " U: " << sp.getParticleEnergy();
    cout << " T: " << sp.getParticleTemperature();
    cout << " P: " << sp.getParticleDynamicalPressure();
    cout << " phi: " << sp.getParticlePhi() << endl;
    // save minimized configuration
    std::string currentDir = outDir + std::to_string(sp.getParticlePhi()) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioSP.saveParticleConfiguration(currentDir);
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << "\nTarget density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      sp.scaleSoftParticles(scaleFactor);
      currentPhi = sp.getParticlePhi();
      cout << "\nNew phi: " << currentPhi << ", average size: " << sp.getMeanParticleSigma() << endl;
      searchStep += 1;
    }
  }
  return 0;
}
