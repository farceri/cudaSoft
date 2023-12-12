//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Include C++ header files

#include "include/SP2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, saveFinal = true, logSave, linSave;
  long numParticles = atol(argv[9]), nDim = 2;
  long maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long initialStep = atof(argv[7]), step = 0, firstDecade = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  double ec = 240, cutDistance = 1, cutoff, maxDelta, sigma, damping, forceUnit, timeUnit;
  double timeStep = atof(argv[2]), inertiaOverDamping = atof(argv[8]), strain = atof(argv[10]);
  double Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-langevin/";
  thrust::host_vector<double> boxSize(nDim);
  dirSample = whichDynamics + "Dr" + argv[4] + "/";
  //dirSample = whichDynamics + "Dr" + argv[4] + "-f0" + argv[5] + "/";
  // initialize sp object
	SP2D sp(numParticles, nDim);
  sp.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  ioSPFile ioSP(&sp);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      logSave = true;
      outDir = outDir + "dynamicsLy" + argv[10] + "-log/";
      //linSave = true;
      //outDir = outDir + "dynamicsLE" + argv[10] + "/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        if(initialStep != 0) {
          inDir = outDir;
        }
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
      //outDir = inDir + "../../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioSP.readParticlePackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioSP.readParticleActiveState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioSP.openEnergyFile(energyFile);
  // initialization
  sp.setEnergyCostant(ec);
  cutoff = (1 + cutDistance) * sp.getMinParticleSigma();
  sigma = sp.getMeanParticleSigma();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = 1 / damping;
  //timeUnit = sigma * sigma * damping;
  timeStep = sp.setTimeStep(timeStep * timeUnit);
  //timeStep = sp.setTimeStep(timeStep);
  forceUnit = inertiaOverDamping / sigma;
  //forceUnit = 1 / (inertiaOverDamping * sigma);
  cout << "Inertia over damping: " << inertiaOverDamping << " damping: " << damping << " sigma: " << sigma << endl;
  cout << "Tinject: " << Tinject << " time step: " << timeStep << " taup: " << timeUnit/Dr << endl;
  cout << "Peclet number: " << driving * forceUnit * timeUnit / (damping * Dr * sigma);
  cout << " f0: " << driving*forceUnit << ", " << driving << " Dr: " << Dr/timeUnit << ", " << Dr << endl;
  driving = driving*forceUnit;
  Dr = Dr/timeUnit;
  ioSP.saveParticleDynamicalParams(outDir, sigma, damping, Dr, driving);
  if(initialStep == 0) {
    sp.applyExtension(strain);
    boxSize = sp.getBoxSize();
    cout << "new box - Lx: " << boxSize[0] << ", Ly: " << boxSize[1] << endl;
  }
  // initialize simulation
  sp.calcParticleNeighborList(cutDistance);
  sp.calcParticleForceEnergy();
  if(initialStep == 0) {
    currentDir = outDir + "/affine/";
    std::experimental::filesystem::create_directory(currentDir);
    ioSP.saveParticlePacking(currentDir);
  }
  sp.initSoftParticleActiveLangevin(Tinject, Dr, driving, damping, readState);
  //waveQ = sp.getSoftWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  while(step != maxStep) {
    sp.softParticleActiveLangevinLoop();
    if(step % checkPointFreq == 0) {
      cout << "shear Active: current step: " << step + initialStep;
      cout << " U/N: " << sp.getParticleEnergy() / numParticles;
      cout << " T: " << sp.getParticleTemperature();
      cout << " pressure: " << sp.getParticleVirialPressure();
      if(step != 0 && updateCount > 0) {
        cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
      } else {
        cout << " no updates" << endl;
      }
      updateCount = 0;
      if(saveFinal == true) {
        ioSP.saveParticlePacking(outDir);
      }
    }
    if(logSave == true) {
      if(step > (multiple * maxStep)) {
        saveFreq = 1;
        multiple += 1;
      }
      if((step - (multiple-1) * maxStep) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * maxStep) % saveFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleActiveState(currentDir);
        ioSP.saveParticleStressEnergy(step+initialStep, timeStep, numParticles);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioSP.saveParticleActiveState(currentDir);
        ioSP.saveParticleStressEnergy(step+initialStep, timeStep, numParticles);
      }
    }
    maxDelta = sp.getParticleMaxDisplacement();
    if(3*maxDelta > cutoff) {
      sp.calcParticleNeighborList(cutDistance);
      sp.resetLastPositions();
      updateCount += 1;
    }
    step += 1;
  }
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioSP.saveParticlePacking(outDir);
  }
  ioSP.closeEnergyFile();

  return 0;
}
