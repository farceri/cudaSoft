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
  bool read = false, readState = false;
  long numParticles = 32, nDim = 2, numVertexPerParticle = 48, numVertices;
  long iteration = 0, maxIterations = 1e06, minStep = 20, numStep = 0;
  long maxSearchStep = 1500, searchStep = 0, saveEnergyFreq = 1e03;
  long step = 0, maxStep = 1e03, updateFreq = 1e02, printFreq = int(maxStep / 10);
  double polydispersity = 0.17, previousPhi, currentPhi, deltaPhi = 1e-03, scaleFactor;
  double cutDistance = 2., forceTollerance = 1e-12, FIREStep = 1e-02;
  double timeStep = 1e-03, phi0 = 0.2, phiTh = 1., Tinject = 8e02;
  double ec = 240, calA0 = 1.;
  std::string outDir = argv[1], currentDir, inDir;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(0, 0, 0, ec);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = outDir + argv[2];
    ioDPM.readRigidPackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioDPM.readRigidState(inDir, numParticles, numVertices, nDim);
    }
  } else {
    // initialize polydisperse packing
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setRandomParticles(phi0, 1.4); //extraRad
    // minimize soft sphere packing
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
    while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      dpm.particleFIRELoop();
      if(iteration % updateFreq == 0) {
        dpm.calcParticleNeighborList(cutDistance);
      }
      iteration += 1;
    }
    cout << "\nFIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << dpm.getParticleEnergy() << endl;
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
  }
  // quasistatic isothermal compression
  dpm.setTimeStep(timeStep);
  // quasistatic thermal compression
  cout << "time step: " << timeStep << endl;
  dpm.calcParticlesShape();
  currentPhi = dpm.getPhi();
  cout << "current phi: " << currentPhi << endl;
  while (searchStep < maxSearchStep) {
    previousPhi = currentPhi;
    dpm.initRigidBrownian(Tinject, cutDistance, readState);
    dpm.calcNeighborList(cutDistance);
    dpm.calcRigidForceEnergy();
    // equilibrate dynamics
    step = 0;
    while(step != maxStep) {
      dpm.rigidBrownianLoop();
      if(step % printFreq == 0) {
        cout << "Rigid Brownian: current step: " << step;
        cout << " energy: " << dpm.getPotentialEnergy() + dpm.getKineticEnergy();
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getVertexISF() << endl;
      }
      if(step % updateFreq == 0) {
        dpm.calcNeighborList(cutDistance);
      }
      step += 1;
    }
    cout << "\nRigid Brownian: current step: " << step;
    cout << " potential energy: " << dpm.getPotentialEnergy();
    cout << " T: " << dpm.getTemperature();
    cout << " pressure: " << dpm.getPressure();
    cout << " phi: " << dpm.getPhi() << endl;
    // save minimized configuration
    std::string currentDir = outDir + std::to_string(dpm.getPhi()).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveRigidPacking(currentDir);
    ioDPM.saveRigidState(currentDir);
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << " target density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((previousPhi + deltaPhi) / previousPhi);
      dpm.scaleVertices(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << " new phi: " << currentPhi << endl;
      searchStep += 1;
    }
  }

  return 0;
}
