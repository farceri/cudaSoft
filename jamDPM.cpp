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
  bool read = false, readCellFormat = false, readState = false; // this is for thermalization
  long numParticles = 32, nDim = 2, numVertexPerParticle = 32, numVertices;
  long iteration = 0, maxIterations = 5e06, printFreq = 1e05;
  long minStep = 20, numStep = 0, updateFreq = 10, step = 0, maxStep = 1e04;
  long maxSearchStep = 500, searchStep = 0, updateNVEFreq = 1e01, saveEnergyFreq = 1e03;
  double polydispersity = 0.17, previousPhi, currentPhi = 0.4, phiTh = 1.2, deltaPhi = 2e-03, scaleFactor;
  double cutDistance = 2., forceTollerance = 1e-12, pressureTollerance = 1e-10, pressure;
  double ea = 1e03, el = 1, eb = 0, ec = 1, timeStep, dt0 = 1., forceCheck;
  double calA0 = 1.1, thetaA = 1., thetaK = 0., Tinject = 1e-02;
  bool jamCheck = 0, overJamCheck = 0, underJamCheck = 0;
  std::string outDir = argv[1], currentDir, inDir, inFile, energyFile;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, dt0, 10*dt0, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // read initial configuration
  if(read == true) {
    if(readCellFormat == true) {
      inFile = "/home/francesco/Documents/Data/isoCompression/poly32-compression.test";
      ioDPM.readPackingFromCellFormat(inFile, 1);
    } else {
      inDir = argv[2];
      inDir = outDir + inDir;
      ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setSinusoidalRestAngles(thetaA, thetaK);
    dpm.setRandomParticles(currentPhi, 1.2); //extraRad
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
  dpm.calcParticlesShape();
  cout << "current packing fraction: " << setprecision(precision) << dpm.getPhi() << endl;
  numVertices = dpm.getNumVertices();
  thrust::host_vector<double> positions(numVertices * nDim);
  thrust::host_vector<double> radii(numVertices);
  thrust::host_vector<double> lengths(numVertices);
  thrust::host_vector<double> areas(numParticles);

  // quasistatic isotropic compression
  timeStep = dpm.setTimeScale(dt0);
  cout << "fire time step: " << timeStep << endl;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> vertexFIREparams = {0.2, 0.5, 1.1, 0.99, timeStep, 10*timeStep, 0.2};
  dpm.initFIRE(vertexFIREparams, minStep, numStep, numVertices);
  currentPhi = dpm.getPhi();
  previousPhi = currentPhi;
  while (searchStep < maxSearchStep) {
    dpm.calcNeighborList(cutDistance);
    dpm.calcForceEnergy();
    iteration = 0;
    forceCheck = dpm.getTotalForceMagnitude();
    while((forceCheck > forceTollerance) && (iteration != maxIterations)) {
      dpm.vertexFIRELoop();
      forceCheck = dpm.getTotalForceMagnitude();
      if(iteration % printFreq == 0) {
        cout << "\nFIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
        cout << " energy: " << dpm.getPotentialEnergy() << endl;
      }
      if(iteration % updateFreq == 0) {
        dpm.calcNeighborList(cutDistance);
      }
      iteration += 1;
    }
    pressure = dpm.getPressure();
    cout << "\nFIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
    cout << " energy: " << dpm.getPotentialEnergy();
    cout << " pressure " << pressure << endl;
    // save minimized configuration
    currentDir = outDir + std::to_string(currentPhi).substr(0,7) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveConfiguration(currentDir);
    // check configuration after energy minimization
    jamCheck = (pressure < 2.0 * pressureTollerance && pressure > pressureTollerance);
    overJamCheck = (pressure > 2.0 * pressureTollerance);
    underJamCheck = (pressure < pressureTollerance);
    if(jamCheck) {
      cout << "Compression step: " << searchStep;
      cout << " Found jammed configuration, pressure: " << setprecision(precision) << pressure;
      cout << " packing fraction: " << currentPhi << endl;
      if(pressureTollerance == 1e-05) {
        searchStep = maxSearchStep; // exit condition
      }
      pressureTollerance = 1e-05;
      deltaPhi = 2e-03;
    } else {
      // compute scaleFactor with binary search
      if(underJamCheck) {
        scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found underjammed configuration, scaleFactor: " << scaleFactor;
        // save most recent underjammed configuration
        previousPhi = currentPhi;
        positions = dpm.getVertexPositions();
        radii = dpm.getVertexRadii();
        lengths = dpm.getRestLengths();
        areas = dpm.getRestAreas();
      } else if(overJamCheck) {
        deltaPhi *= 0.5;
        scaleFactor = sqrt((previousPhi + deltaPhi) / previousPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found overjammed configuration, scaleFactor: " << scaleFactor;
        // copy back most recent underjammed configuration and compress half much
        dpm.setVertexPositions(positions);
        dpm.setVertexRadii(radii);
        dpm.setRestLengths(lengths);
        dpm.setRestAreas(areas);
      }
      cout << " potential energy: " << dpm.getPotentialEnergy() << endl;
      if(currentPhi > phiTh) {
        // run NVE integrator
        cout << "\nNVE thermalization" << endl;
        // thermalize packing after each energy minimization
        dpm.initNVE(Tinject, readState);
        while(step != maxStep) {
          dpm.NVELoop();
          if(step % saveEnergyFreq == 0) {
            ioDPM.saveEnergy(step, timeStep);
            if(step % printFreq == 0) {
              cout << "NVE: current step: " << step;
              cout << " potential energy: " << dpm.getPotentialEnergy();
              cout << " T: " << dpm.getTemperature();
              cout << " phi: " << dpm.getPhi() << endl;
            }
          }
          if(step % updateNVEFreq == 0) {
            dpm.calcNeighborList(cutDistance);
          }
          step += 1;
        }
        step = 0;
      }
      dpm.scaleVertices(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << " new packing fraction: " << currentPhi << endl;
      searchStep += 1;
    }
  }

  return 0;
}
