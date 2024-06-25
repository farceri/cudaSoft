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
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << endl;
  }

  void saveParticleWorkEnergy(long step, double timeStep, long numParticles, double driving, double tp, double width) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    std::tuple<double, double> work = sp_->getParticleWork(width);
    energyFile << setprecision(precision) << get<0>(work) << "\t";
    energyFile << setprecision(precision) << get<1>(work) << "\t";
    std::tuple<double, double> activeWork = sp_->getParticleActiveWork(driving, tp, width);
    energyFile << setprecision(precision) << get<0>(activeWork) << "\t";
    energyFile << setprecision(precision) << get<1>(activeWork) << endl;
  }

  void saveParticleNoseHooverEnergy(long step, double timeStep, long numParticles) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    double mass, damping;
    sp_->getNoseHooverParams(mass, damping);
    energyFile << setprecision(precision) << damping << endl;
  }

  void saveParticleDoubleNoseHooverEnergy(long step, double timeStep, long numParticles, long num1) {
    double epot = sp_->getParticlePotentialEnergy();
    std::tuple<double, double, double> ekins = sp_->getParticleKineticEnergy12();
    double etot = epot + get<2>(ekins);
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << get<0>(ekins) / num1 << "\t";
    energyFile << setprecision(precision) << get<1>(ekins) / (numParticles - num1) << "\t";
    energyFile << setprecision(precision) << get<2>(ekins) / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    double mass, damping1, damping2;
    sp_->getDoubleNoseHooverParams(mass, damping1, damping2);
    energyFile << setprecision(precision) << damping1 << "\t";
    energyFile << setprecision(precision) << damping2 << endl;
  }

  void saveParticleDoubleEnergy(long step, double timeStep, long numParticles, long num1) {
    double epot = sp_->getParticlePotentialEnergy() / numParticles;
    std::tuple<double, double, double> ekins = sp_->getParticleKineticEnergy12();
    double etot = epot + get<2>(ekins);
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << get<0>(ekins) / num1 << "\t";
    energyFile << setprecision(precision) << get<1>(ekins) / (numParticles - num1) << "\t";
    energyFile << setprecision(precision) << get<2>(ekins) / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << endl;
  }

  void saveParticleWallEnergy(long step, double timeStep, long numParticles, double range) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleWallForce(range, 0.0) << endl;
  }

  void saveParticleCenterWallEnergy(long step, double timeStep, long numParticles, double range, double width) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleWallForce(range, width) << endl;
  }

  void saveParticleWallStressEnergy(long step, double timeStep, long numParticles, double range) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleWallForceUpDown(range) << "\t";
    std::tuple<double, double, double> stress = sp_->getParticleStressComponents();
    energyFile << setprecision(precision) << get<0>(stress) << "\t";
    energyFile << setprecision(precision) << get<1>(stress) << "\t";
    energyFile << setprecision(precision) << get<2>(stress) << endl;
  }

  void saveParticleActiveWallEnergy(long step, double timeStep, long numParticles, double range, double driving) {
    double epot = sp_->getParticlePotentialEnergy();
    double ekin = sp_->getParticleKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot / numParticles << "\t";
    energyFile << setprecision(precision) << ekin / numParticles << "\t";
    energyFile << setprecision(precision) << etot / numParticles << "\t";
    energyFile << setprecision(precision) << sp_->getParticleActiveWallForce(range, driving) << "\t";
    energyFile << setprecision(precision) << sp_->getTotalParticleWallCount() << endl;
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

  void save2DIndexFile(string fileName, thrust::host_vector<long> data, long numCols) {
    this->openOutputFile(fileName);
    long numRows = int(data.size()/numCols);
    for (long row = 0; row < numRows; row++) {
      for(long col = 0; col < numCols; col++) {
        outputFile << data[row * numCols + col] << "\t";
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

  void savePackingParams(string dirName) {
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    long nDim = sp_->getNDim();
    long numParticles = sp_->getNumParticles();
    saveParams << "numParticles" << "\t" << numParticles << endl;
    saveParams << "nDim" << "\t" << nDim << endl;
    saveParams << "sigma" << "\t" << 2 * sp_->getMeanParticleSigma() << endl;
    saveParams << "epsilon" << "\t" << sp_->getEnergyCostant() << endl;
    saveParams << "dt" << "\t" << sp_->dt << endl;
    saveParams << "phi" << "\t" << sp_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << sp_->getParticleEnergy() / numParticles << endl;
    saveParams << "temperature" << "\t" << sp_->getParticleTemperature() << endl;
    saveParams.close();
  }

  void saveParticlePacking(string dirName) {
    savePackingParams(dirName);
    // save vectors
    long nDim = sp_->getNDim();
    save1DFile(dirName + "boxSize.dat", sp_->getBoxSize());
    save1DFile(dirName + "particleRad.dat", sp_->getParticleRadii());
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    if(sp_->simControl.particleType == simControlStruct::particleEnum::active) {
      if(nDim == 2) {
        save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
      } else if(nDim == 3) {
        save2DFile(dirName + "particleAngles.dat", sp_->getParticleAngles(), nDim);
      } else {
        cout << "FileIO::saveParticlePacking: only dimensions 2 and 3 are allowed for particleAngles!" << endl;
      }
    }
    //save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
    //save2DFile(dirName + "particleForces.dat", sp_->getParticleForces(), sp_->nDim);
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

  void readParticleVelocity(string dirName, long numParticles_, long nDim_) {
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

  void readParticleState(string dirName, long numParticles_, long nDim_) {
    readParticleVelocity(dirName, numParticles_, nDim_);
    if(sp_->simControl.particleType == simControlStruct::particleEnum::active) {
      thrust::host_vector<double> particleAngle_(numParticles_);
      if(nDim_ == 2) {
        particleAngle_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
      } else if(nDim_ == 3) {
        particleAngle_.resize(numParticles_ * nDim_);
        particleAngle_ = read3DFile(dirName + "particleAngles.dat", numParticles_);
      } else {
        cout << "FileIO::readParticleState: only dimensions 2 and 3 are allowed for particleAngles!" << endl;
      }
      sp_->setParticleAngles(particleAngle_);
    }
  }

  void saveParticleState(string dirName) {
    save2DFile(dirName + "particlePos.dat", sp_->getParticlePositions(), sp_->nDim);
    save2DFile(dirName + "particleVel.dat", sp_->getParticleVelocities(), sp_->nDim);
    if(sp_->simControl.particleType == simControlStruct::particleEnum::active) {
      if(sp_->nDim == 2) {
        save1DFile(dirName + "particleAngles.dat", sp_->getParticleAngles());
      } else {
        save2DFile(dirName + "particleAngles.dat", sp_->getParticleAngles(), sp_->nDim);
      }
    }
  }

  void saveParticleEnergies(string dirName) {
    save1DFile(dirName + "particleEnergies.dat", sp_->getParticleEnergies());
  }

  void saveParticleContacts(string dirName) {
    sp_->calcParticleContacts(0.);
    save2DFile(dirName + "particleContacts.dat", sp_->getContacts(), sp_->contactLimit);
  }

  void saveParticleNeighbors(string dirName) {
    if(sp_->simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
      save2DIndexFile(dirName + "particleNeighbors.dat", sp_->getParticleNeighbors(), sp_->partNeighborListSize);
    }
  }

  void saveLangevinParams(string dirName, double damping) {
    string fileParams = dirName + "dynParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "damping" << "\t" << damping << endl;
    if(sp_->simControl.particleType == simControlStruct::particleEnum::active) {
      double driving, taup;
      sp_->getSelfPropulsionParams(driving, taup);
      saveParams << "taup" << "\t" << taup << endl;
      saveParams << "f0" << "\t" << driving << endl;
    }
    saveParams.close();
  }

  void saveNoseHooverParams(string dirName) {
    double mass, damping;
    sp_->getNoseHooverParams(mass, damping);
    string fileParams = dirName + "nhParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "mass" << "\t" << mass << endl;
    saveParams << "damping" << "\t" << damping << endl;
    if(sp_->simControl.particleType == simControlStruct::particleEnum::active) {
      double driving, taup;
      sp_->getSelfPropulsionParams(driving, taup);
      saveParams << "taup" << "\t" << taup << endl;
      saveParams << "f0" << "\t" << driving << endl;
    }
    saveParams.close();
  }

  void readNoseHooverParams(string dirName, double &mass, double &damping) {
    string fileParams = dirName + "nhParams.dat";
    ifstream readParams(fileParams.c_str());
    if (!readParams.is_open()) {
      cout << "Error: Unable to open file " << fileParams << " - setting default values" << endl;
      return;
    }
    string paramName;
    double paramValue;
    while (readParams >> paramName >> paramValue) {
      if(paramName == "mass") {
        mass = paramValue;
      } else if(paramName == "damping") {
        damping = paramValue;
      }
    }
    readParams.close();
    if(mass == 1 && damping == 1) {
      cout << "FileIO::saveNoseHooverParams: mass and damping are not saved in nhParams.dat! Setting mass and damping to 1" << endl;
    }
  }

  void saveDoubleNoseHooverParams(string dirName) {
    double mass, damping1, damping2;
    sp_->getDoubleNoseHooverParams(mass, damping1, damping2);
    string fileParams = dirName + "nhParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "mass" << "\t" << mass << endl;
    saveParams << "damping1" << "\t" << damping1 << endl;
    saveParams << "damping2" << "\t" << damping2 << endl;
    saveParams.close();
  }

  void readDoubleNoseHooverParams(string dirName, double &mass, double &damping1, double &damping2) {
    string fileParams = dirName + "nhParams.dat";
    ifstream readParams(fileParams.c_str());
    if (!readParams.is_open()) {
      cout << "Error: Unable to open file " << fileParams << " - setting default values" << endl;
      return;
    }
    string paramName;
    double paramValue;
    while (readParams >> paramName >> paramValue) {
      if(paramName == "mass") {
        mass = paramValue;
      } else if(paramName == "damping1") {
        damping1 = paramValue;
      } else if(paramName == "damping2") {
        damping2 = paramValue;
      }
    }
    readParams.close();
    if(mass == 1 && damping1 == 1 && damping2 == 1) {
      cout << "FileIO::saveDoubleNoseHooverParams: mass and damping are not saved in nhParams.dat! Setting mass, damping1 and damping2 to 1" << endl;
    }
  }

  void saveFlowParams(string dirName, double sigma, double damping, double gravity, double viscosity, double ew) {
    string fileParams = dirName + "flowParams.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "sigma" << "\t" << sigma << endl;
    saveParams << "damping" << "\t" << damping << endl;
    saveParams << "gravity" << "\t" << gravity << endl;
    saveParams << "viscosity" << "\t" << viscosity << endl;
    saveParams << "ew" << "\t" << ew << endl;
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
      outputFile << "ITEM: ATOMS id type radius xu yu zu" << endl;
    } else {
      outputFile << "ITEM: ATOMS id type radius xu yu" << endl;
    }
    int type = 1;
    for (long particleId = 0; particleId < numParticles_; particleId++) {
      if(particleId < sp_->num1) {
        type = 1;
      } else {
        type = 2;
      }
      //outputFile << particleId + 1 << "\t" << 1 << "\t" << particleId + 1 << "\t";
      //outputFile << rad[particleId] << "\t" << pos[particleId * nDim] << "\t" << pos[particleId * nDim + 1] << "\t" << pos[particleId * nDim + 2] << endl;
      outputFile << particleId + 1 << "\t" << type << "\t" << rad[particleId] << "\t";
      if(nDim_ == 3) {
        outputFile << pos[particleId * nDim_] << "\t" << pos[particleId * nDim_ + 1] << "\t" << pos[particleId * nDim_ + 2] << endl;
      } else {
        outputFile << pos[particleId * nDim_] << "\t" << pos[particleId * nDim_ + 1] << endl;
      }
    }
  }

};

#endif // FILEIO_H //
