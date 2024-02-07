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
  long printFreq = int(maxStep / 10), updateCount = 0;
  double polydispersity = 0.2, previousPhi, currentPhi, deltaPhi = 5e-02, scaleFactor, isf = 1;
  double LJcut = 5.5, cutDistance = 1, forceTollerance = 1e-08, waveQ, FIREStep = 1e-02, dt = atof(argv[2]);
  double ec = 1, ew = 100, Tinject = atof(argv[3]), damping, inertiaOverDamping = 10, phi0 = 0.2, phiTh = 0.7;
  double timeStep, timeUnit, sigma, cutoff, maxDelta, lx = atof(argv[4]), gravity = 9.8e-04;
  std::string currentDir, outDir = argv[1], inDir;
  thrust::host_vector<double> boxSize(nDim);
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setGeometryType(simControlStruct::geometryEnum::normal);
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = argv[6];
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
    sp.setDisplacementCutoff(cutoff, cutDistance);
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
  sp.setGravityType(simControlStruct::gravityEnum::on);
  sp.setGravity(gravity, ew);
  //sp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
  //sp.setLJcutoff(LJcut);
  // quasistatic thermal compression
  currentPhi = sp.getParticlePhi();
  cout << "current phi: " << currentPhi << ", average size: " << sigma << endl;
  previousPhi = currentPhi;
  // initilize velocities only the first time
  while (searchStep < maxSearchStep) {
    timeUnit = sigma / sqrt(ec);
    damping = (inertiaOverDamping / sigma);
    timeStep = sp.setTimeStep(dt*timeUnit);
    cout << "Time step: " << timeStep << ", damping: " << damping << endl;
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    sp.initSoftParticleLangevin(Tinject, damping, readState);
    cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
    sp.setDisplacementCutoff(cutoff, cutDistance);
    sp.resetUpdateCount();
    sp.setInitialPositions();
    waveQ = sp.getSoftWaveNumber();
    // equilibrate dynamics
    step = 0;
    while(step != maxStep) {
      sp.softParticleLangevinLoop();
      if(step % printFreq == 0) {
        isf = sp.getParticleISF(waveQ);
        cout << "Langevin: current step: " << step;
        cout << " U/N: " << sp.getParticleEnergy() / numParticles;
        cout << " K/N: " << sp.getParticleKineticEnergy() / numParticles;
        cout << " ISF: " << sp.getParticleISF(waveQ);
        updateCount = sp.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << printFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        sp.resetUpdateCount();
      }
      step += 1;
    }
    cout << "Final step - T: " << sp.getParticleTemperature();
    cout << " P: " << sp.getParticleDynamicalPressure();
    cout << " phi: " << sp.getParticlePhi() << endl;
    // save minimized configuration
    currentDir = outDir + std::to_string(sp.getParticlePhi()).substr(0,5) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioSP.saveParticlePacking(currentDir);
    // check if target density is met
    if(currentPhi > phiTh) {
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
