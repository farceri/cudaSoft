//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/SP2D.h"
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
  long numParticles = 32, nDim = 3; // this is just a default
  long iteration = 0, maxIterations = 1e06, printFreq = 1e04;
  long minStep = 20, numStep = 0, updateFreq = 1e02;
  double polydispersity = 0.17, phi0 = atof(argv[2]), dt0 = 1e-03;
  double ec = 1, cutDistance = 1, forceTollerance = 1e-10, forceCheck;
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, dt0, 10*dt0, 0.2};
  std::string inFile, outDir = argv[1];
	// initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  std::experimental::filesystem::create_directory(outDir);
  // initialize polydisperse packing
  sp.setPolyRandomSoftParticles(phi0, polydispersity);
  sp.setEnergyCostant(ec); //kc = 1
  // minimize soft sphere packing
  sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
  sp.setParticleMassFIRE();
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  ioSP.saveParticleState(outDir);
  while((sp.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
    sp.particleFIRELoop();
    if(iteration % updateFreq == 0) {
      sp.calcParticleNeighborList(cutDistance);
    }
    if(iteration % printFreq == 0) {
      cout << "\nFIRE: iteration: " << iteration;
      cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
      cout << " energy: " << setprecision(precision) << sp.getParticleEnergy() << endl;
    }
    iteration += 1;
  }
  cout << "\nFIRE: iteration: " << iteration;
  cout << " maxUnbalancedForce: " << setprecision(precision) << sp.getParticleMaxUnbalancedForce();
  cout << " energy: " << setprecision(precision) << sp.getParticleEnergy() << endl;
  ioSP.saveParticlePacking(outDir);
  cout << "saved packing with packing fraction: " << setprecision(precision) << sp.get3DParticlePhi() << endl;

  return 0;
}
