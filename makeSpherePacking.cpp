//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/DPM3D.h"
#include "include/FileIO.h"
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
  long numParticles = 32, nDim = 3, numVertexPerParticle = 32; // this is just a default
  long iteration = 0, maxIterations = 1e06, printFreq = 1e04;
  long minStep = 20, numStep = 0, updateFreq = 1e02;
  double polydispersity = 0.17, phi0 = atof(argv[2]), dt0 = 1e-03;
  double ec = 1, cutDistance = 1, forceTollerance = 1e-10, forceCheck;
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, dt0, 10*dt0, 0.2};
  std::string inFile, outDir = argv[1];
	// initialize dpm object
	DPM3D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // initialize polydisperse packing
  dpm.setPolyRandomSoftParticles(phi0, polydispersity);
  dpm.setEnergyCosts(0, 0, 0, ec); //kc = 1
  // minimize soft sphere packing
  dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
  dpm.setParticleMassFIRE();
  dpm.calcParticleNeighborList(cutDistance);
  dpm.calcParticleForceEnergy();
  ioDPM.saveParticleState(outDir);
  while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
    dpm.particleFIRELoop();
    if(iteration % updateFreq == 0) {
      dpm.calcParticleNeighborList(cutDistance);
    }
    if(iteration % printFreq == 0) {
      cout << "\nFIRE: iteration: " << iteration;
      cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
      cout << " energy: " << setprecision(precision) << dpm.getParticleEnergy() << endl;
    }
    iteration += 1;
  }
  cout << "\nFIRE: iteration: " << iteration;
  cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
  cout << " energy: " << setprecision(precision) << dpm.getParticleEnergy() << endl;
  ioDPM.saveParticlePacking(outDir);
  cout << "saved packing with packing fraction: " << setprecision(precision) << dpm.getParticlePhi() << endl;

  return 0;
}
