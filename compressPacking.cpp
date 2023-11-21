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
  bool read = false, readState = false;
  long numParticles = atol(argv[5]), nDim = 2;
  long iteration = 0, maxIterations = 1e05, minStep = 20, numStep = 0;
  long maxStep = 1e04, step = 0, maxSearchStep = 1500, searchStep = 0;
  long printFreq = int(maxStep / 10), updateFreq = 10;
  double polydispersity = 0.2, previousPhi, currentPhi, deltaPhi = 4e-02, scaleFactor, isf = 1;
  double cutDistance = 1.5, forceTollerance = 1e-08, waveQ, FIREStep = 1e-02, dt = atof(argv[2]);
  double ec = 1, Tinject = atof(argv[3]), damping, inertiaOverDamping = 10, phi0 = 0.2, phiTh = 0.5;
  double timeStep, timeUnit, escale = 1, sigma, cutoff, maxDelta, lx = atof(argv[4]);
  std::string currentDir, outDir = argv[1], inDir;
  thrust::host_vector<double> boxSize(nDim);
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[4];
    inDir = outDir + inDir;
    ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
    sigma = sp.getMeanParticleSigma();
    sp.setEnergyCostant(ec);
    cutoff = cutDistance * sp.getMinParticleSigma();
    if(readState == true) {
      ioSP.readParticleState(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing
    sp.setScaledPolyRandomSoftParticles(phi0, polydispersity, lx);
    sp.scaleParticlePacking();
    //sp.scaleParticlesAndBoxSize();
    sigma = sp.getMeanParticleSigma();
    sp.setEnergyCostant(ec);
    cutoff = cutDistance * sp.getMinParticleSigma();
    sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    sp.setParticleMassFIRE();
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    cout << "Generate initial configurations with FIRE \n" << endl;
    while((sp.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      sp.particleFIRELoop();
      if(iteration % printFreq == 0 && iteration != 0) {
        cout << "FIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
        cout << " energy: " << sp.getParticleEnergy() << endl;
      }
      maxDelta = sp.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
      }
      iteration += 1;
    }
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << sp.getParticleEnergy() << "\n" << endl;
    //currentDir = outDir + "initial/";
    //std::experimental::filesystem::create_directory(currentDir);
    //ioSP.saveParticlePacking(currentDir);
  }
  // quasistatic thermal compression
  sp.setPotentialType(simControlStruct::potentialEnum::WCA);
  currentPhi = sp.getParticlePhi();
  cout << "current phi: " << currentPhi << ", average size: " << sigma << endl;
  previousPhi = currentPhi;
  // initilize velocities only the first time
  while (searchStep < maxSearchStep) {
    damping = sqrt(inertiaOverDamping) / sigma;
    timeUnit = 1 / damping;
    timeStep = sp.setTimeStep(dt * timeUnit);
    cout << "Time step: " << timeStep << ", damping: " << damping << endl;
    sp.initSoftParticleLangevin(Tinject, damping, readState);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    waveQ = sp.getSoftWaveNumber();
    sp.setInitialPositions();
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
        cout << " U: " << sp.getParticleEnergy() / numParticles;
        cout << " K: " << sp.getParticleKineticEnergy() / numParticles;
        cout << " ISF: " << isf << endl;
      }
      maxDelta = sp.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
      }
      step += 1;
    }
    cout << "Langevin: current step: " << step;
    cout << " U: " << sp.getParticleEnergy();
    cout << " T: " << sp.getParticleTemperature();
    cout << " P: " << sp.getParticleDynamicalPressure();
    cout << " phi: " << sp.getParticlePhi() << endl;
    // save minimized configuration
    currentDir = outDir + std::to_string(sp.getParticlePhi()) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioSP.saveParticleConfiguration(currentDir);
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << "\nTarget density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      sp.scaleParticles(scaleFactor);
      sp.scaleParticlePacking();
      boxSize = sp.getBoxSize();
      currentPhi = sp.getParticlePhi();
      cout << "\nNew phi: " << currentPhi << " Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
      searchStep += 1;
    }
  }
  return 0;
}
