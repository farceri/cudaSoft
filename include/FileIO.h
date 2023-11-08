//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//

#ifndef FILEIO_H
#define FILEIO_H

#include "SP2D.h"
#include "defs.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

class ioSPFile
{
public:
  ifstream inputFile;
  ofstream outputFile;
  ofstream energyFile;
  ofstream corrFile;
  SP2D * sp_;

  ioSPFile() = default;
  ioSPFile(SP2D * spPtr) {
    this->sp_ = spPtr;
  }

  // open file and check if it throws an error
  void openInputFile(string fileName) {
    inputFile = ifstream(fileName.c_str());
    if (!inputFile.is_open()) {
      cerr << "ioSPFile::openInputFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void openOutputFile(string fileName) {
    outputFile = ofstream(fileName.c_str());
    if (!outputFile.is_open()) {
      cerr << "ioSPFile::openOutputFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void openEnergyFile(string fileName) {
    energyFile = ofstream(fileName.c_str());
    if (!energyFile.is_open()) {
      cerr << "ioSPFile::openEnergyFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void saveParticleSimpleEnergy(long step, double timeStep, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticleEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
  }

  void saveParticleEnergy(long step, double timeStep, double waveNumber, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticleEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleVirialPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleDynamicalPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleISF(waveNumber) << endl;
  }

  void saveParticleStressStrain(double strain, long numParticles) {
    energyFile << strain << "\t";
    energyFile << setprecision(precision) << sp_->getParticleEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleVirialPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleDynamicalPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleShearStress() << endl;
  }

  void saveParticleStressEnergy(long step, double timeStep, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticleEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleVirialPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleDynamicalPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleShearStress() << endl;
  }

  void saveParticleActiveEnergy(long step, double timeStep, double waveNumber, double driving, double numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticleEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleVirialPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleDynamicalPressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleISF(waveNumber) << endl;
  }

  void closeEnergyFile() {
    energyFile.close();
  }

  thrust::host_vector<double> read1DIndexFile(string fileName, long numRows) {
    thrust::host_vector<long> data;
    this->openInputFile(fileName);
    string inputString;
    long tmp;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%ld", &tmp);
      data.push_back(tmp);
    }
    inputFile.close();
    return data;
  }

  void save1DIndexFile(string fileName, thrust::host_vector<long> data) {
    this->openOutputFile(fileName);
    long numRows = data.size();
    for (long row = 0; row < numRows; row++) {
      //sprintf(outputFile, "%ld \n", data[row]);
      outputFile << setprecision(precision) << data[row] << endl;
    }
    outputFile.close();
  }


  thrust::host_vector<double> read1DFile(string fileName, long numRows) {
    thrust::host_vector<double> data;
    this->openInputFile(fileName);
    string inputString;
    double tmp;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%lf", &tmp);
      data.push_back(tmp);
    }
    inputFile.close();
    return data;
  }

  void save1DFile(string fileName, thrust::host_vector<double> data) {
    this->openOutputFile(fileName);
    long numRows = data.size();
    for (long row = 0; row < numRows; row++) {
      //sprintf(outputFile, "%lf \n", data[row]);
      outputFile << setprecision(precision) << data[row] << endl;
    }
    outputFile.close();
  }

  thrust::host_vector<double> read2DFile(string fileName, long numRows) {
    thrust::host_vector<double> data;
    this->openInputFile(fileName);
    string inputString;
    double data1, data2;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%lf %lf", &data1, &data2);
      data.push_back(data1);
      data.push_back(data2);
    }
    inputFile.close();
    return data;
  }

  void save2DFile(string fileName, thrust::host_vector<double> data, long numCols) {
    this->openOutputFile(fileName);
    long numRows = int(data.size()/numCols);
    for (long row = 0; row < numRows; row++) {
      for(long col = 0; col < numCols; col++) {
        outputFile << setprecision(precision) << data[row * numCols + col] << "\t";
      }
      outputFile << endl;
    }
    outputFile.close();
  }

  void readParticlePackingFromDirectory(string dirName, long numParticles_, long nDim_) {
    sp_->initParticleVariables(numParticles_);
    sp_->initParticleNeighbors(numParticles_);
    sp_->syncParticleNeighborsToDevice();
    thrust::host_vector<double> boxSize_(nDim_);
    thrust::host_vector<double> pPos_(numParticles_ * nDim_);
    thrust::host_vector<double> pRad_(numParticles_);

    boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
    sp_->setBoxSize(boxSize_);
    pPos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    sp_->setParticlePositions(pPos_);
    pRad_ = read1DFile(dirName + "particleRad.dat", numParticles_);
    sp_->setParticleRadii(pRad_);
    // set length scales
    sp_->setLengthScaleToOne();
    boxSize_ = sp_->getBoxSize();
    cout << "FileIO::readParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << endl;
  }

  void readPBCParticlePackingFromDirectory(string dirName, long numParticles_, long nDim_) {
    sp_->initParticleVariables(numParticles_);
    sp_->initParticleNeighbors(numParticles_);
    sp_->syncParticleNeighborsToDevice();
    thrust::host_vector<double> boxSize_(nDim_);
    thrust::host_vector<double> pPos_(numParticles_ * nDim_);
    thrust::host_vector<double> pRad_(numParticles_);

    boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
    sp_->setBoxSize(boxSize_);
    pPos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    sp_->setPBCParticlePositions(pPos_);
    pRad_ = read1DFile(dirName + "particleRad.dat", numParticles_);
    sp_->setParticleRadii(pRad_);
    // set length scales
    sp_->setLengthScaleToOne();
    cout << "FileIO::readPBCParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << endl;
  }

  void saveAthermalParticlePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << sp_->getNumParticles() << endl;
    saveParams << "dt" << "\t" << sp_->dt << endl;
    saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << sp_->getParticleEnergy() / sp_->getNumParticles() << endl;
    saveParams << "pressure" << "\t" << sp_->getParticleVirialPressure() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
    save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void saveParticlePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << sp_->getNumParticles() << endl;
    saveParams << "dt" << "\t" << sp_->dt << endl;
    saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << sp_->getParticleEnergy() / sp_->getNumParticles() << endl;
    saveParams << "temperature" << "\t" << sp_->getParticleTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
    save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void readParticleState(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
    particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
    sp_->setParticleVelocities(particleVel_);
  }

  void readParticleActiveState(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<double> particleAngle_(numParticles_);
    thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
    particleAngle_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
    sp_->setParticleAngles(particleAngle_);
    particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
    sp_->setParticleVelocities(particleVel_);
  }

  void saveParticleState(string dirName) {
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
    //sp_->calcParticleContacts(0.);
    //save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
    //save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
  }

  void saveParticleActiveState(string dirName) {
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void saveParticleAttractiveState(string dirName) {
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
    save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
  }

  void saveParticleDynamicalParams(string dirName, double sigma, double damping, double Dr, double driving) {
    string fileParams = dirName + "dynParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "pressure" << "\t" << sp_->getParticleDynamicalPressure() << endl;
    saveParams << "sigma" << "\t" << sigma << endl;
    saveParams << "damping" << "\t" << damping << endl;
    saveParams << "Dr" << "\t" << Dr << endl;
    saveParams << "f0" << "\t" << driving << endl;
    saveParams.close();
  }

  void saveParticleContacts(string dirName) {
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
    save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
    //save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    //save2DFile(dirName + "lastPos.dat", sp_->getLastPositions(), sp_->nDim);
  }

  void saveParticleConfiguration(string dirName) {
    saveParticlePacking(dirName);
    saveParticleState(dirName);
  }

  void saveParticleActiveConfiguration(string dirName) {
    saveParticlePacking(dirName);
    saveParticleActiveState(dirName);
  }

  void saveParticleAttractiveConfiguration(string dirName) {
    saveParticlePacking(dirName);
    saveParticleAttractiveState(dirName);
  }

};

#endif // FILEIO_H //
