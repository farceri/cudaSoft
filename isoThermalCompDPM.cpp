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
  bool read = true, readState = true;
  long numParticles = 512, nDim = 2, numVertexPerParticle = 32, numVertices;
  long iteration = 0, maxIterations = 5e06, minStep = 20, numStep = 0;
  long maxStep = 1e04, step = 0, maxSearchStep = 1500, searchStep = 0;
  long printFreq = int(maxStep / 10), updateFreq = 10;
  long softStep = 1e04, printSoft = int(softStep / 10), updateSoft = 10;
  double polydispersity = 0.2, previousPhi, currentPhi = 0.2, deltaPhi = 2e-03, scaleFactor;
  double cutDistance = 1, forceTollerance = 1e-12, waveQ, FIREStep = 1e-02, dtSoft = 1e-02, dt = 1e-03;
  double Tsoft = 1e-03, Tinject = atof(argv[2]), phiTh = 1.;
  double sigma, damping, inertiaOverDamping = 100, timeStep, timeUnit;
  double ea = 1000, el = 1, eb = 5e-02, ec = 1, calA0 = 1.2, thetaA = 1., thetaK = 0.;
  // kc and kl has to be equal
  // the k's are actually epsilon's
  std::string outDir = argv[1], currentDir, inDir, energyFile;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // read initial configuration
  if(read == true) {
    inDir = argv[3];
    inDir = outDir + inDir;
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
    }
    sigma = dpm.getMeanParticleSize();
    dpm.setEnergyCosts(ea, el, eb, ec);
  } else {
    // initialize polydisperse packing
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setSinusoidalRestAngles(thetaA, thetaK);
    dpm.setRandomParticles(currentPhi, 1.2); //extraRad
    sigma = dpm.getMeanParticleSigma();
    dpm.setEnergyCosts(0, 0, 0, ec);
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
    // equilibrate soft particle packing to make the voids homogeneous
    damping = sqrt(inertiaOverDamping * ec) / sigma;
    timeUnit = sigma * sigma * damping / ec; // epsilon_c is 1
    timeStep = dpm.setTimeStep(timeUnit * dtSoft);
    cout << "\nEquilibrate soft particles, time step: " << timeStep << endl;
    step = 0;
    // thermalize packing after each energy minimization
    dpm.initSoftParticleLangevin(Tsoft, damping, readState);
    while(step != softStep) {
      dpm.softParticleLangevinLoop();
      if(step % printSoft == 0) {
        cout << "Langevin: current step: " << step;
        cout << " E: " << dpm.getSmoothPotentialEnergy() + dpm.getParticleKineticEnergy();
        cout << " T: " << dpm.getParticleTemperature();
        cout << " pressure: " << dpm.getParticlePressure() << endl;
      }
      if(step % updateSoft == 0) {
        dpm.calcParticleNeighborList(cutDistance);
      }
      step += 1;
    }
    currentDir = outDir + "sp/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
    sigma = dpm.getMeanParticleSize();
    dpm.setEnergyCosts(ea, el, eb, ec);
    cout << "Spring constants: area " << ea/sigma << " segment " << el/sigma << " bending " << eb/sigma << " interaction " << ec/sigma << endl;
    currentDir = outDir + "dpm/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.savePacking(currentDir);
  }
  numVertices = dpm.getNumVertices();
  dpm.calcParticlesShape();
  currentPhi = dpm.getPhi();
  cout << "Current packing fraction: " << currentPhi << endl;
  previousPhi = currentPhi;
  // isotropic isothermal compression
  while (searchStep < maxSearchStep) {
    sigma = dpm.getMeanParticleSize();
    damping = sqrt(inertiaOverDamping * ec) / sigma;
    cout << "damping: " << damping << " with inertia over damping: " << inertiaOverDamping << endl;
    timeUnit = sigma * sigma * damping / ec;
    timeStep = dpm.setTimeStep(timeUnit * dt);
    //timeStep = dpm.setTimeStep(dt);
    cout << "\nTime step: " << timeStep << endl;
    dpm.initLangevin(Tinject, damping, readState);
    dpm.calcNeighborList(cutDistance);
    dpm.calcForceEnergy();
    waveQ = dpm.getDeformableWaveNumber();
    // equilibrate dynamics
    step = 0;
    while(step != maxStep) {
      dpm.langevinLoop();
      if(step % printFreq == 0) {
        cout << "Langevin: current step: " << step;
        cout << " E: " << dpm.getPotentialEnergy() + dpm.getKineticEnergy();
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi() << endl;
      }
      if(step % updateFreq == 0) {
        dpm.calcNeighborList(cutDistance);
      }
      step += 1;
    }
    cout << "Langevin: final step: " << step;
    cout << " energy: " << dpm.getPotentialEnergy();
    cout << " T: " << dpm.getTemperature();
    cout << " pressure " << dpm.getPressure();
    cout << " phi: " << dpm.getPhi() << endl;
    // save minimized configuration
    currentDir = outDir + std::to_string(dpm.getPhi()).substr(0,7) + "/";
    //currentDir = outDir + "test/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveConfiguration(currentDir);
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << "Target density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      dpm.scaleVertices(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << "New packing fraction: " << currentPhi << endl;
      searchStep += 1;
    }
    ioDPM.saveEnergy(step, timeStep);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
