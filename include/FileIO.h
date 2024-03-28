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
  ofstream memoryFile;
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

  void reopenEnergyFile(string fileName) {
    energyFile = ofstream(fileName.c_str(), std::fstream::app);
    if (!energyFile.is_open()) {
      cerr << "ioSPFile::openEnergyFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void openMemoryFile(string fileName) {
    memoryFile = ofstream(fileName.c_str());
    if (!memoryFile.is_open()) {
      cerr << "ioSPFile::openMemoryFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void saveMemoryUsage(long step) {
    memoryFile << step + 1 << "\t";
    memoryFile << setprecision(12) << sp_->checkGPUMemory() << endl;
  }

  void saveElapsedTime(double elapsedTime) {
    memoryFile << "Elapsed time - ms:" << setprecision(12) << elapsedTime << " min: " << elapsedTime / (1000*60) << " hr: " << elapsedTime / (1000*60*60) << endl;
  }

  void closeMemoryFile() {
    memoryFile.close();
  }

  void saveParticleSimpleEnergy(long step, double timeStep, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << endl;
  }

  void saveParticleEnergy(long step, double timeStep, double waveNumber, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleISF(waveNumber) << endl;
  }

  void saveParticleStressEnergy(long step, double timeStep, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePressure() << "\t";
    //energyFile << setprecision(precision) << sp_->getParticleShearStress() << endl;
    energyFile << setprecision(precision) << sp_->getParticleExtensileStress() << endl;
  }

  void saveParticleFixedBoxEnergy(long step, double timeStep, long numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePressure() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleWallPressure() << endl;
  }

  void saveParticleWallEnergy(long step, double timeStep, long numParticles, double range) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleWallForce(range) << "\t";
    energyFile << setprecision(precision) << sp_->getTotalParticleWallCount() << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePressure() << endl;
  }

  void saveParticleActiveWallEnergy(long step, double timeStep, long numParticles, double range, double driving) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleActiveWallForce(range, driving) << "\t";
    energyFile << setprecision(precision) << sp_->getTotalParticleWallCount() << "\t";
    energyFile << setprecision(precision) << sp_->getParticleExtensileStress() << endl;
  }

  void saveParticleActiveEnergy(long step, double timeStep, double waveNumber, double driving, double numParticles) {
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePotentialEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleKineticEnergy() / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticlePressure() << "\t";
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


//////////////////////////// write this function in array form ////////////////////////////
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

  thrust::host_vector<double> read3DFile(string fileName, long numRows) {
    thrust::host_vector<double> data;
    this->openInputFile(fileName);
    string inputString;
    double data1, data2, data3;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%lf %lf %lf", &data1, &data2, &data3);
      data.push_back(data1);
      data.push_back(data2);
      data.push_back(data3);
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

  thrust::host_vector<double> readBoxSize(string dirName, long nDim_) {
    thrust::host_vector<double> boxSize_(nDim_);
    boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
    cout << "FileIO::readBoxSize: " << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << endl;
    return boxSize_;
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
    if(nDim_ == 2) {
      pPos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    } else if(nDim_ == 3) {
      pPos_ = read3DFile(dirName + "particlePos.dat", numParticles_);
    } else {
      cout << "FileIO::readParticlePackingFromDirectory: only dimensions 2 and 3 are allowed!" << endl;
    }
    sp_->setParticlePositions(pPos_);
    pRad_ = read1DFile(dirName + "particleRad.dat", numParticles_);
    sp_->setParticleRadii(pRad_);
    // set length scales
    sp_->setLengthScaleToOne();
    boxSize_ = sp_->getBoxSize();
    if(nDim_ == 2) {
      cout << "FileIO::readParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << endl;
    } else if(nDim_ == 3) {
      cout << "FileIO::readParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << ", Lz: " << boxSize_[2] << endl;
    }
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
    if(nDim_ == 2) {
      pPos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    } else if(nDim_ == 3) {
      pPos_ = read3DFile(dirName + "particlePos.dat", numParticles_);
    } else {
      cout << "FileIO::readPBCParticlePackingFromDirectory: only dimensions 2 and 3 are allowed!" << endl;
    }
    sp_->setPBCParticlePositions(pPos_);
    pRad_ = read1DFile(dirName + "particleRad.dat", numParticles_);
    sp_->setParticleRadii(pRad_);
    // set length scales
    sp_->setLengthScaleToOne();
    if(nDim_ == 2) {
      cout << "FileIO::readPBCParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << endl;
    } else if(nDim_ == 3) {
      cout << "FileIO::readPBCParticlePackingFromDirectory: phi: " << sp_->getParticlePhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << ", Lz: " << boxSize_[2] << endl;
    }
  }

  void saveAthermalParticlePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << sp_->getNumParticles() << endl;
    saveParams << "dt" << "\t" << sp_->dt << endl;
    saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << sp_->getParticlePotentialEnergy() / sp_->getNumParticles() << endl;
    saveParams << "pressure" << "\t" << sp_->getParticlePressure() << endl;
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
    saveParams << "energy" << "\t" << sp_->getParticlePotentialEnergy() / sp_->getNumParticles() << endl;
    saveParams << "temperature" << "\t" << sp_->getParticleTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
    save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
    //save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
    //save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    save2DFile(dirName + "particleForces.dat", sp_->getParticleForces(), sp_->nDim);
    //save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
    //sp_->calcParticleContacts(0.);
    //save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void saveParticleActivePacking(string dirName) {
  // save scalars
  string fileParams = dirName + "params.dat";
  ofstream saveParams(fileParams.c_str());
  openOutputFile(fileParams);
  saveParams << "numParticles" << "\t" << sp_->getNumParticles() << endl;
  saveParams << "dt" << "\t" << sp_->dt << endl;
  saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
  saveParams << "energy" << "\t" << sp_->getParticlePotentialEnergy() / sp_->getNumParticles() << endl;
  saveParams << "temperature" << "\t" << sp_->getParticleTemperature() << endl;
  saveParams.close();
  // save vectors
  save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
  save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
  save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
  //save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
  save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
  save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
  save2DFile(dirName + "particleForces.dat", sp_->getParticleForces(), sp_->nDim);
  //save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
  //sp_->calcParticleContacts(0.);
  //save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
}

  void savePBCParticlePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << sp_->getNumParticles() << endl;
    saveParams << "dt" << "\t" << sp_->dt << endl;
    saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << sp_->getParticlePotentialEnergy() / sp_->getNumParticles() << endl;
    saveParams << "temperature" << "\t" << sp_->getParticleTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
    save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
    save2DFile(dirName + "particlePos.dat", sp_->getPBCParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
  }

  void readParticleState(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
    if(nDim_ == 2) {
      particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
    } else if(nDim_ == 3) {
      particleVel_ = read3DFile(dirName + "particleVel.dat", numParticles_);
    } else {
      cout << "FileIO::readParticleState: only dimensions 2 and 3 are allowed!" << endl;
    }
    sp_->setParticleVelocities(particleVel_);
  }

  void readParticleActiveState(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<double> particleAngle_(numParticles_);
    thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
    particleAngle_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
    sp_->setParticleAngles(particleAngle_);
    if(nDim_ == 2) {
      particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
    } else if(nDim_ == 3) {
      particleVel_ = read3DFile(dirName + "particleVel.dat", numParticles_);
    } else {
      cout << "FileIO::readParticleActiveState: only dimensions 2 and 3 are allowed!" << endl;
    }
    sp_->setParticleVelocities(particleVel_);
  }

  void saveParticleState(string dirName) {
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
  }

  void saveParticleActiveState(string dirName) {
    save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
  }

  void saveParticleContacts(string dirName) {
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void saveParticleNeighbors(string dirName) {
    save2DFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
  }

  void saveParticleDynamicalParams(string dirName, double sigma, double damping, double Dr, double driving) {
    string fileParams = dirName + "dynParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "sigma" << "\t" << sigma << endl;
    saveParams << "damping" << "\t" << damping << endl;
    saveParams << "Dr" << "\t" << Dr << endl;
    saveParams << "f0" << "\t" << driving << endl;
    saveParams.close();
  }

  void saveXYZPacking(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<double> pos(numParticles_ * nDim_), rad(numParticles_);
    pos = sp_->getPBCParticlePositions();
    rad = sp_->getParticleRadii();
    long numCols = nDim_;
    this->openOutputFile(dirName + "packing.xyz");
    long numRows = int(pos.size()/numCols);
    outputFile << numParticles_ << "\n\n";
    for (long row = 0; row < numRows; row++) {
      for(long col = 0; col < numCols; col++) {
        outputFile << setprecision(precision) << pos[row * numCols + col] << "\t";
      }
      outputFile << rad[row * numCols + numCols] << endl;
    }
    outputFile.close();
  }

  void saveDumpPacking(string dirName, long numParticles_, long nDim_, long timeStep_) {
    thrust::host_vector<double> pos(numParticles_ * nDim_), rad(numParticles_), boxSize(nDim_);
    pos = sp_->getPBCParticlePositions();
    rad = sp_->getParticleRadii();
    boxSize = sp_->getBoxSize();
    this->openOutputFile(dirName + "packing.dump");
    outputFile << "ITEM: TIMESTEP" << endl;
    outputFile << timeStep_ << endl;
    outputFile << "ITEM: NUMBER OF ATOMS" << endl;
    outputFile << numParticles_ << endl;
    if(nDim_ == 3) {
      outputFile << "ITEM: BOX BOUNDS pp pp fixed" << endl;
    } else {
      outputFile << "ITEM: BOX BOUNDS pp pp" << endl;
    }
    outputFile << 0 << "\t" << boxSize[0] << endl;
    outputFile << 0 << "\t" << boxSize[1] << endl;
    if(nDim_ == 3) {
      outputFile << 0 << "\t" << boxSize[2] << endl;
      outputFile << "ITEM: ATOMS id radius xu yu zu" << endl;
    } else {
      outputFile << "ITEM: ATOMS id radius xu yu" << endl;
    }
    for (long particleId = 0; particleId < numParticles_; particleId++) {
      //outputFile << particleId + 1 << "\t" << 1 << "\t" << particleId + 1 << "\t";
      //outputFile << rad[particleId] << "\t" << pos[particleId * nDim] << "\t" << pos[particleId * nDim + 1] << "\t" << pos[particleId * nDim + 2] << endl;
      outputFile << particleId + 1 << "\t" << rad[particleId] << "\t";
      if(nDim_ == 3) {
        outputFile << pos[particleId * nDim_] << "\t" << pos[particleId * nDim_ + 1] << "\t" << pos[particleId * nDim_ + 2] << endl;
      } else {
        outputFile << pos[particleId * nDim_] << "\t" << pos[particleId * nDim_ + 1] << endl;
      }
    }
  }

};

#endif // FILEIO_H //
