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
  bool save = true, saveSame = false;
  long iteration = 0, maxIteration = 1e04, printFreq = maxIteration;
  long numParticles = 8192, nDim = 2, minStep = 20, numStep = 0, repetition = 0;
  double FIREStep = 1e-04, newFIREStep, phi, pressure, cutoff, maxDelta, strain, strainStep = atof(argv[3]);
  double cutDistance = 1, polydispersity = 0.20, ec = 1, sigma, maxStrain = atof(argv[2]);
  double forceCheck, lastEnergyCheck, energyCheck, forceTollerance = 1e-12, energyTollerance = 1e-05;
  std::string inDir = argv[1], outDir, currentDir, dirSample = "shear5/", saveFile;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize sp object
	SP2D sp(numParticles, nDim);
  ioSPFile ioSP(&sp);
  if(saveSame == true) {
    outDir = inDir;
  } else {
    outDir = inDir + dirSample;
  }
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  // output file
  saveFile = outDir + "output.dat";
  ioSP.openOutputFile(saveFile);
  sp.setEnergyCostant(ec);
  phi = sp.getParticlePhi();
  sigma = sp.getMeanParticleSigma();
  FIREStep = FIREStep * sigma;
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  strain = strainStep;
  sp.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  sp.initFIRE(particleFIREparams, minStep, numStep, numParticles);
  sp.setParticleMassFIRE();
  while (strain < (maxStrain + strainStep)) {
    sp.setLEshift(strain);
    sp.applyLEShear(strainStep);
    sp.calcParticleNeighborList(cutDistance);
    sp.calcParticleForceEnergy();
    iteration = 0;
    newFIREStep = FIREStep;
    forceCheck = sp.getParticleMaxUnbalancedForce();
    energyCheck = sp.getParticleEnergy();
    lastEnergyCheck = energyCheck;
    forceTollerance /= sp.getMeanParticleSigma();
    while((energyCheck > energyTollerance) && (iteration != maxIteration)) {
      sp.particleFIRELoop();
      forceCheck = sp.getParticleMaxUnbalancedForce();
      energyCheck = sp.getParticleEnergy() / numParticles;
      if(iteration % printFreq == 0) {
        cout << "FIRE: iteration: " << iteration;
        cout << " force: " << setprecision(precision) << forceCheck;
        cout << " energy: " << setprecision(precision) << energyCheck;
        cout << " pressure: " << sp.getParticleVirialPressure();
        cout << " shear stress: " << sp.getParticleShearStress() << endl;
        if(lastEnergyCheck <= energyCheck) {
          repetition += 1;
        }
        if(repetition > 3) {
          newFIREStep /= 2;
          sp.setTimeStepFIRE(newFIREStep);
          //cout << "Dividing the time step in half" << endl;
          repetition = 0;
        }
        lastEnergyCheck = energyCheck;
      }
      maxDelta = sp.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        sp.calcParticleNeighborList(cutDistance);
        sp.resetLastPositions();
      }
      iteration += 1;
    }
    cout << "FIRE: iteration: " << iteration;
    cout << " force: " << setprecision(precision) << forceCheck;
    cout << " energy: " << setprecision(precision) << energyCheck;
    cout << " pressure: " << sp.getParticleVirialPressure();
    cout << " shear stress: " << sp.getParticleShearStress() << endl;
    // save minimized configuration
    if(save == true) {
      std::string currentDir = outDir + "strain" + std::to_string(strain) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioSP.saveAthermalParticlePacking(currentDir);
    }
    ioSP.saveParticleStress(strain, numParticles);
    strain += strainStep;
  }

  return 0;
}
