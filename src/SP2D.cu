//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// FUNCTION DECLARATIONS

#include "../include/SP2D.h"
#include "../include/cudaKernel.cuh"
#include "../include/Simulator.h"
#include "../include/FIRE.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

using namespace std;
using std::cout;
using std::endl;

//************************** sp object definition ***************************//
SP2D::SP2D(long nParticles, long dim) {
  // default values
  srand48(time(0));
  dimBlock = 256;
  nDim = dim;
  numParticles = nParticles;
  // the default is monodisperse size distribution
  setDimBlock(dimBlock);
  setNDim(nDim);
  setNumParticles(numParticles);
	simControl.geometryType = simControlStruct::geometryEnum::normal;
	simControl.potentialType = simControlStruct::potentialEnum::harmonic;
	syncSimControlToDevice();
  // default parameters
  dt = 1e-04;
  rho0 = 1;
	ec = 1;
	l1 = 0;
	l2 = 0;
  LEshift = 0;
  cutDistance = 1;
  updateCount = 0;
  d_boxSize.resize(nDim);
  thrust::fill(d_boxSize.begin(), d_boxSize.end(), double(1));
  d_stress.resize(nDim * nDim);
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  // particle variables
  initParticleVariables(numParticles);
  initParticleDeltaVariables(numParticles);
  // initialize contacts and neighbors
  initContacts(numParticles);
  initParticleNeighbors(numParticles);
  syncParticleNeighborsToDevice();
}

SP2D::~SP2D() {
	// clear all vectors and pointers
	d_boxSize.clear();
  d_stress.clear();
  d_particleRad.clear();
  d_particlePos.clear();
  d_particleVel.clear();
  d_particleForce.clear();
  d_particleEnergy.clear();
  d_particleAngle.clear();
  // delta variables
  d_particleInitPos.clear();
  d_particleLastPos.clear();
  d_particleDelta.clear();
  d_particleDisp.clear();
  // contacts and neighbors
  d_contactList.clear();
  d_numContacts.clear();
  d_contactVectorList.clear();
  d_partNeighborList.clear();
  d_partMaxNeighborList.clear();
}

void SP2D::initParticleVariables(long numParticles_) {
  d_particleRad.resize(numParticles_);
  d_particlePos.resize(numParticles_ * nDim);
  d_particleVel.resize(numParticles_ * nDim);
  d_particleForce.resize(numParticles_ * nDim);
  d_particleEnergy.resize(numParticles_);
  d_particleAngle.resize(numParticles_);
  thrust::fill(d_particleRad.begin(), d_particleRad.end(), double(0));
  thrust::fill(d_particlePos.begin(), d_particlePos.end(), double(0));
  thrust::fill(d_particleVel.begin(), d_particleVel.end(), double(0));
  thrust::fill(d_particleForce.begin(), d_particleForce.end(), double(0));
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  thrust::fill(d_particleAngle.begin(), d_particleAngle.end(), double(0));
}

void SP2D::initParticleDeltaVariables(long numParticles_) {
  d_particleInitPos.resize(numParticles_ * nDim);
  d_particleLastPos.resize(numParticles * nDim);
  d_particleDelta.resize(numParticles_ * nDim);
  d_particleDisp.resize(numParticles_);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  thrust::fill(d_particleLastPos.begin(), d_particleLastPos.end(), double(0));
  thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
  thrust::fill(d_particleDisp.begin(), d_particleDisp.end(), double(0));
}

void SP2D::initContacts(long numParticles_) {
  long maxContacts = 8 * nDim; // guess
  d_numContacts.resize(numParticles_);
  d_contactList.resize(numParticles_ * maxContacts);
  d_partNeighborList.resize(numParticles_ * maxContacts);
  d_contactVectorList.resize(numParticles_ * nDim * maxContacts);
  thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
  thrust::fill(d_contactList.begin(), d_contactList.end(), double(0));
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), double(0));
  thrust::fill(d_contactVectorList.begin(), d_contactVectorList.end(), double(0));
}

void SP2D::initParticleNeighbors(long numParticles_) {
  partNeighborListSize = 0;
  partMaxNeighbors = 0;
  d_partNeighborList.resize(numParticles_);
  d_partMaxNeighborList.resize(numParticles_);
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), partMaxNeighbors);
}

//**************************** setters and getters ***************************//
//send simControl information to the gpu
void SP2D::syncSimControlToDevice() {
	cudaMemcpyToSymbol(d_simControl, &simControl, sizeof(simControl));
}

//get simControl information from the gpu
void SP2D::syncSimControlFromDevice() {
	cudaMemcpyFromSymbol(&simControl, d_simControl, sizeof(simControl));
}

void SP2D::setGeometryType(simControlStruct::geometryEnum geometryType_) {
	simControl.geometryType = geometryType_;
  if(simControl.geometryType == simControlStruct::geometryEnum::normal) {
    cout << "SP2D: setGeometryType: geometryType: normal" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::leesEdwards) {
    cout << "SP2D: setGeometryType: geometryType: leesEdwards" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedBox) {
    cout << "SP2D: setGeometryType: geometryType: fixedBox" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides) {
    cout << "SP2D: setGeometryType: geometryType: leesSides" << endl;
  } else {
    cout << "SP2D: setGeometryType: please specify valid geometryType: normal, leesEdwards, fixedBox or fixedSides" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::geometryEnum SP2D::getGeometryType() {
	syncSimControlFromDevice();
	return simControl.geometryType;
}

void SP2D::setPotentialType(simControlStruct::potentialEnum potentialType_) {
	simControl.potentialType = potentialType_;
  if(simControl.potentialType == simControlStruct::potentialEnum::harmonic) {
    cout << "SP2D: setPotentialType: potentialType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::lennardJones) {
    cout << "SP2D: setPotentialType: potentialType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::WCA) {
    cout << "SP2D: setPotentialType: potentialType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::adhesive) {
    cout << "SP2D: setPotentialType: potentialType: adhesive" << endl;
  } else {
    cout << "SP2D: setPotentialType: please specify valid potentialType: harmonic, lennardJones, WCA or adhesive" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::potentialEnum SP2D::getPotentialType() {
	syncSimControlFromDevice();
	return simControl.potentialType;
}

bool SP2D::testSimControlSync() {
	bool returnValue = true;
	simControlStruct temp = simControl;
	syncSimControlFromDevice();
	returnValue = ((temp.geometryType == simControl.geometryType) &&
                  (temp.potentialType == simControl.potentialType));
	if (returnValue == true) {
    cout << "SP2D::testSimControlSync: symControl is in sync" << endl;
	}
	if (returnValue == false) {
		cout << "SP2D::testSimControlSync: symControl is out of sync" << endl;
	}
  cout << "geometryType = " << (temp.geometryType == simControl.geometryType) << endl;
  cout << "potentialType = " << (temp.potentialType == simControl.potentialType) << endl;
	simControl = temp;
	syncSimControlToDevice();
	return returnValue;
}

void SP2D::setLEshift(double LEshift_) {
	syncSimControlFromDevice();
	if(simControl.geometryType == simControlStruct::geometryEnum::leesEdwards) {
		LEshift = LEshift_;
		cudaError err = cudaMemcpyToSymbol(d_LEshift, &LEshift, sizeof(LEshift));
		if(err != cudaSuccess) {
			cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
		}
	}
	else {
		cout << "SP2D::setLEshift: attempting to set LEshift without using LE boundary conditions" << endl;
	}
  cout << "SP2D::setLEshift: LEshift: " << LEshift << endl;
}

double SP2D::getLEshift() {
  double LEshiftFromDevice;
	cudaError err = cudaMemcpyFromSymbol(&LEshiftFromDevice, d_LEshift, sizeof(d_LEshift));
	if(err != cudaSuccess) {
		cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
	}
	return LEshiftFromDevice;
}

void SP2D::applyLEShear(double LEshift_) {
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto shearPosition = [=] __device__ (long particleId) {
		double shearPos;
		shearPos = pPos[particleId * d_nDim] + LEshift_ * pPos[particleId * d_nDim + 1];
		shearPos -= round(shearPos / boxSize[0]) * boxSize[0];
		pPos[particleId * d_nDim] = shearPos;
	};

	thrust::for_each(r, r+numParticles, shearPosition);
}

void SP2D::applyExtension(double shifty_) {
  // first set the new boxSize
  thrust::host_vector<double> newBoxSize(nDim);
  newBoxSize = getBoxSize();
  newBoxSize[1] = (1 + shifty_) * newBoxSize[1];
  setBoxSize(newBoxSize);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + shifty_) * pPos[particleId * d_nDim + 1];
		extendPos -= round(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyLinearExtension(thrust::host_vector<double> &newBoxSize_, double shifty_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + shifty_) * pPos[particleId * d_nDim + 1];
		extendPos -= round(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, double shiftx_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto biaxialPosition = [=] __device__ (long particleId) {
		double extendPos, compressPos;
		extendPos = (1 + shifty_) * pPos[particleId * d_nDim + 1];
		extendPos -= round(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
		compressPos = (1 + shiftx_) * pPos[particleId * d_nDim];
		compressPos -= round(compressPos / boxSize[0]) * boxSize[0];
		pPos[particleId * d_nDim] = compressPos;
	};

	thrust::for_each(r, r+numParticles, biaxialPosition);
}

void SP2D::applyLinearCompression(thrust::host_vector<double> &newBoxSize_, double shiftx_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + shiftx_) * pPos[particleId * d_nDim];
		extendPos -= round(extendPos / boxSize[0]) * boxSize[0];
		pPos[particleId * d_nDim] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}


// TODO: add error checks for all the getters and setters
void SP2D::setDimBlock(long dimBlock_) {
	dimBlock = dimBlock_;
	dimGrid = (numParticles + dimBlock - 1) / dimBlock;
  cudaError err = cudaMemcpyToSymbol(d_dimBlock, &dimBlock, sizeof(dimBlock));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
  err = cudaMemcpyToSymbol(d_dimGrid, &dimGrid, sizeof(dimGrid));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
}

long SP2D::getDimBlock() {
  long dimBlockFromDevice;
  //dimBlockFromDevice = d_dimBlock;
  cudaError err = cudaMemcpyFromSymbol(&dimBlockFromDevice, d_dimBlock, sizeof(d_dimBlock));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
  if (dimBlock != dimBlockFromDevice) {
    cout << "DPM::getDimBlock: dimBlock on host does not match dimBlock on device" << endl;
  }
	return dimBlockFromDevice;
}

void SP2D::setNDim(long nDim_) {
  nDim = nDim_;
  cudaMemcpyToSymbol(d_nDim, &nDim, sizeof(nDim));
}

long SP2D::getNDim() {
  long nDimFromDevice;
  cudaMemcpyFromSymbol(&nDimFromDevice, d_nDim, sizeof(d_nDim));
	return nDimFromDevice;
}

void SP2D::setNumParticles(long numParticles_) {
  numParticles = numParticles_;
  cudaMemcpyToSymbol(d_numParticles, &numParticles, sizeof(numParticles));
}

long SP2D::getNumParticles() {
  long numParticlesFromDevice;
  cudaMemcpyFromSymbol(&numParticlesFromDevice, d_numParticles, sizeof(d_numParticles));
	return numParticlesFromDevice;
}

void SP2D::setParticleLengthScale() {
  rho0 = thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>())/numParticles; // set dimensional factor
  cout << " lengthscale: " << rho0 << endl;
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

void SP2D::setLengthScaleToOne() {
  rho0 = 1.; // for soft particles
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

//TODO: error messages for all the vector getters and setters
void SP2D::setBoxSize(thrust::host_vector<double> &boxSize_) {
  if(boxSize_.size() == ulong(nDim)) {
    d_boxSize = boxSize_;
    double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
    cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  } else {
    cout << "SP2D::setBoxSize: size of boxSize does not match nDim" << endl;
  }
}

thrust::host_vector<double> SP2D::getBoxSize() {
  thrust::host_vector<double> boxSizeFromDevice;
  if(d_boxSize.size() == ulong(nDim)) {
    cudaMemcpyFromSymbol(&d_boxSize, d_boxSizePtr, sizeof(d_boxSizePtr));
    boxSizeFromDevice = d_boxSize;
  } else {
    cout << "SP2D::getBoxSize: size of boxSize from device does not match nDim" << endl;
  }
  return boxSizeFromDevice;
}

void SP2D::setParticleRadii(thrust::host_vector<double> &particleRad_) {
  d_particleRad = particleRad_;
}

thrust::host_vector<double> SP2D::getParticleRadii() {
  thrust::host_vector<double> particleRadFromDevice;
  particleRadFromDevice = d_particleRad;
  return particleRadFromDevice;
}

double SP2D::getMeanParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>()) / numParticles;
}

double SP2D::getMinParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(1), thrust::minimum<double>());
}

void SP2D::setParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
}

void SP2D::setPBCParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
  // check pbc
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  kernelCheckParticlePBC<<<dimGrid, dimBlock>>>(pPosPBC, pPos);
  // copy to device
  d_particlePos = d_particlePosPBC;
}

thrust::host_vector<double> SP2D::getParticlePositions() {
  thrust::host_vector<double> particlePosFromDevice;
  particlePosFromDevice = d_particlePos;
  return particlePosFromDevice;
}

thrust::host_vector<double> SP2D::getPBCParticlePositions() {
  // check pbc
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  kernelCheckParticlePBC<<<dimGrid, dimBlock>>>(pPosPBC, pPos);
  // copy to host
  thrust::host_vector<double> particlePosFromDevice;
  particlePosFromDevice = d_particlePosPBC;
  return particlePosFromDevice;
}

void SP2D::resetLastPositions() {
  d_particleLastPos = getParticlePositions();
}

void SP2D::setInitialPositions() {
  d_particleInitPos = getParticlePositions();
}

thrust::host_vector<double> SP2D::getLastPositions() {
  thrust::host_vector<double> lastPosFromDevice;
  lastPosFromDevice = d_particleLastPos;
  return lastPosFromDevice;
}

void SP2D::setParticleVelocities(thrust::host_vector<double> &particleVel_) {
  d_particleVel = particleVel_;
}

thrust::host_vector<double> SP2D::getParticleVelocities() {
  thrust::host_vector<double> particleVelFromDevice;
  particleVelFromDevice = d_particleVel;
  return particleVelFromDevice;
}

void SP2D::setParticleForces(thrust::host_vector<double> &particleForce_) {
  d_particleForce = particleForce_;
}

thrust::host_vector<double> SP2D::getParticleForces() {
  thrust::host_vector<double> particleForceFromDevice;
  particleForceFromDevice = d_particleForce;
  return particleForceFromDevice;
}

thrust::host_vector<double> SP2D::getParticleEnergies() {
  thrust::host_vector<double> particleEnergyFromDevice;
  particleEnergyFromDevice = d_particleEnergy;
  return particleEnergyFromDevice;
}

void SP2D::setParticleAngles(thrust::host_vector<double> &particleAngle_) {
  d_particleAngle = particleAngle_;
}

thrust::host_vector<double> SP2D::getParticleAngles() {
  thrust::host_vector<double> particleAngleFromDevice;
  particleAngleFromDevice = d_particleAngle;
  return particleAngleFromDevice;
}

thrust::host_vector<long> SP2D::getContacts() {
  thrust::host_vector<long> contactListFromDevice;
  contactListFromDevice = d_contactList;
  return contactListFromDevice;
}

void SP2D::printContacts() {
  for (long particleId = 0; particleId < numParticles; particleId++) {
    cout << "particleId: " << particleId << " list of contacts: ";
    for (long contactId = 0; contactId < d_numContacts[particleId]; contactId++) {
      cout << d_contactList[particleId * contactLimit + contactId] << " ";
    }
    cout << endl;
  }
}

double SP2D::getParticlePhi() {
  thrust::device_vector<double> d_radSquared(numParticles);
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radSquared.begin(), square());
  return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / (d_boxSize[0] * d_boxSize[1]);
}

double SP2D::get3DParticlePhi() {
  thrust::device_vector<double> d_radCubed(numParticles);
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radCubed.begin(), cube());
  return thrust::reduce(d_radCubed.begin(), d_radCubed.end(), double(0), thrust::plus<double>()) * 3 * PI / (4 * d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
}

double SP2D::getParticleMSD() {
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelCalcParticleDistanceSq<<<dimGrid,dimBlock>>>(particlePos, particleInitPos, particleDelta);
  return thrust::reduce(d_particleDelta.begin(), d_particleDelta.end(), double(0), thrust::plus<double>()) / (numParticles * d_boxSize[0] * d_boxSize[1]);
}

double SP2D::getParticleMaxDisplacement() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, pDisp);
  return thrust::reduce(d_particleDisp.begin(), d_particleDisp.end(), double(-1), thrust::maximum<double>());
}

void SP2D::setDisplacementCutoff(double cutoff_, double cutDistance_) {
  cutoff = cutoff_;
  cutDistance = cutDistance_;
  cout << "DPM2D::setDisplacementCutoff - cutoff: " << cutoff << " cutDistance: " << cutDistance << endl;
}

void SP2D::resetUpdateCount() {
  updateCount = double(0);
  //cout << "DPM2D::resetUpdateCount - updatCount " << updateCount << endl;
}

long SP2D::getUpdateCount() {
  return updateCount;
}

void SP2D::checkParticleMaxDisplacement() {
  double maxDelta;
  maxDelta = getParticleMaxDisplacement();
  if(3*maxDelta > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta: " << maxDelta << " cutoff: " << cutoff << endl;
  }
}

double SP2D::getSoftWaveNumber() {
  if(nDim == 2) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getParticlePhi() / (PI * numParticles)));
  } else if(nDim == 3) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * get3DParticlePhi() / (PI * numParticles)));
  } else {
    cout << "SP2D::getSoftWaveNumber: this function works only for dim = 2 and 3" << endl;
    return 0;
  }
}

double SP2D::getParticleISF(double waveNumber_) {
  thrust::device_vector<double> d_particleSF(numParticles);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleSF = thrust::raw_pointer_cast(&d_particleSF[0]);
  kernelCalcParticleScatteringFunction<<<dimGrid,dimBlock>>>(particlePos, particleInitPos, particleSF, waveNumber_);
  return thrust::reduce(d_particleSF.begin(), d_particleSF.end(), double(0), thrust::plus<double>()) / numParticles;
}

//************************ initialization functions **************************//
void SP2D::setPolyRandomSoftParticles(double phi0, double polyDispersity) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale, boxLength = 1.;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = exp(mean + randNum * sigma);
  }
  scale = sqrt(getParticlePhi() / phi0);
  for (long dim = 0; dim < nDim; dim++) {
    boxSize[dim] = boxLength;
  }
  setBoxSize(boxSize);
  // extract random positions
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] /= scale;
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
  }
  // need to set this otherwise forces are zeros
  //setParticleLengthScale();
  setLengthScaleToOne();
}

//************************ initialization functions **************************//
void SP2D::setScaledPolyRandomSoftParticles(double phi0, double polyDispersity, double lx) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = exp(mean + randNum * sigma);
  }
  boxSize[0] = lx;
  boxSize[1] = 1;
  setBoxSize(boxSize);
  scale = sqrt(getParticlePhi() / phi0);
  boxSize[0] = lx * scale;
  boxSize[1] = scale;
  setBoxSize(boxSize);
  // extract random positions
  for (long particleId = 0; particleId < numParticles; particleId++) {
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
  }
  // need to set this otherwise forces are zeros
  //setParticleLengthScale();
  setLengthScaleToOne();
}

void SP2D::set3DPolyRandomSoftParticles(double phi0, double polyDispersity) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale, boxLength = 1.;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  cout << "ok1" << endl;
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = exp(mean + randNum * sigma);
  }
  scale = cbrt(get3DParticlePhi() / phi0);
  for (long dim = 0; dim < nDim; dim++) {
    boxSize[dim] = boxLength;
  }
  setBoxSize(boxSize);
  // extract random positions
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] /= scale;
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
  }
  // need to set this otherwise forces are zeros
  //setParticleLengthScale();
  setLengthScaleToOne();
}

void SP2D::pressureScaleParticles(double pscale) {
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(pscale), d_particlePos.begin(), thrust::multiplies<double>());
  thrust::transform(d_boxSize.begin(), d_boxSize.end(), thrust::make_constant_iterator(pscale), d_boxSize.begin(), thrust::multiplies<double>());
}

void SP2D::scaleParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
}

void SP2D::scaleParticlePacking() {
  double sigma = getMeanParticleSigma();
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(sigma), d_particleRad.begin(), thrust::divides<double>());
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(sigma), d_particlePos.begin(), thrust::divides<double>());
  thrust::host_vector<double> boxSize_(nDim);
  boxSize_ = getBoxSize();
  for (long dim = 0; dim < nDim; dim++) {
    boxSize_[dim] /= sigma;
  }
  d_boxSize = boxSize_;
  double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
  cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  //setParticleLengthScale();
}

void SP2D::scaleParticleVelocity(double scale) {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), thrust::make_constant_iterator(scale), d_particleVel.begin(), thrust::multiplies<double>());
}

// compute particle angles from velocity
void SP2D::computeParticleAngleFromVel() {
  long p_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

  auto computeParticleAngle = [=] __device__ (long particleId) {
    pAngle[particleId] = atan(pVel[particleId * p_nDim + 1] / pVel[particleId * p_nDim]);
  };

  thrust::for_each(r, r + numParticles, computeParticleAngle);
}

//*************************** force and energy *******************************//
void SP2D::setEnergyCostant(double ec_) {
  ec = ec_;
  cudaMemcpyToSymbol(d_ec, &ec, sizeof(ec));
}

void SP2D::setAttractionConstants(double l1_, double l2_) {
  l1 = l1_;
  l2 = l2_;
  cudaMemcpyToSymbol(d_l1, &l1, sizeof(l1));
  cudaMemcpyToSymbol(d_l2, &l2, sizeof(l2));
}

void SP2D::setLJcutoff(double LJcutoff_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  LJecut = 4 * (1 / pow(LJcutoff, 12) - 1 / pow(LJcutoff, 6));
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
  //cout << "SP2D::setLJcutoff - LJcutoff: " << LJcutoff << " LJecut: " << LJecut << endl;
}

double SP2D::setTimeStep(double dt_) {
  dt = dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

void SP2D::calcParticleForceEnergy() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void SP2D::calcParticleBoxForceEnergy() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
  kernelCalcParticleBoxInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void SP2D::calcParticleSidesForceEnergy() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
  kernelCalcParticleSidesInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

 void SP2D::makeExternalParticleForce(double externalForce) {
   // extract +-1 random forces
   d_particleDelta.resize(numParticles);
   thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
   thrust::counting_iterator<long> index_sequence_begin(lrand48());
   thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleDelta.begin(), randInt(0,1));
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(2), d_particleDelta.begin(), thrust::multiplies<double>());
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(1), d_particleDelta.begin(), thrust::minus<double>());
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(externalForce), d_particleDelta.begin(), thrust::multiplies<double>());
 }

 void SP2D::addExternalParticleForce() {
   long p_nDim(nDim);
   auto r = thrust::counting_iterator<long>(0);
 	 double *pDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
 	 double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

   auto addExternalForce = [=] __device__ (long particleId) {
     pForce[particleId * p_nDim] += pDelta[particleId];
   };

   thrust::for_each(r, r + numParticles, addExternalForce);
 }

 thrust::host_vector<double> SP2D::getExternalParticleForce() {
   // return signed external forces
   thrust::host_vector<double> particleExternalForce;
   particleExternalForce = d_particleDelta;
   return particleExternalForce;
 }

 void SP2D::addConstantParticleForce(double externalForce, long maxIndex) {
   long p_nDim(nDim);
   double p_externalForce(externalForce);
   auto r = thrust::counting_iterator<long>(0);
 	 double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

   auto addExternalForce = [=] __device__ (long particleId) {
     pForce[particleId * p_nDim] += p_externalForce;
   };

   thrust::for_each(r, r + maxIndex, addExternalForce);
 }

 // return the sum of force magnitudes
 double SP2D::getParticleTotalForceMagnitude() {
   thrust::device_vector<double> forceSquared(d_particleForce.size());
   thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
   // sum squares
   double totalForceMagnitude = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(0), thrust::plus<double>()) / (numParticles * nDim));
   forceSquared.clear();
   return totalForceMagnitude;
 }

double SP2D::getParticleMaxUnbalancedForce() {
  thrust::device_vector<double> forceSquared(d_particleForce.size());
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  double maxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
  return maxUnbalancedForce;
}

void SP2D::calcParticleStressTensor() {
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pStress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcParticleStressTensor<<<dimGrid, dimBlock>>>(pRad, pPos, pStress);
}

double SP2D::getParticleVirialPressure() {
   calcParticleStressTensor();
	 return (d_stress[0] + d_stress[3]) / (nDim * d_boxSize[0] * d_boxSize[1]);
}

double SP2D::getParticleShearStress() {
   calcParticleStressTensor();
	 return (d_stress[1] + d_stress[2]) / (nDim * d_boxSize[0] * d_boxSize[1]);
}

double SP2D::getParticleExtensileStress() {
   calcParticleStressTensor();
	 return d_stress[3] / (d_boxSize[0] * d_boxSize[1]);
}

double SP2D::getParticleWallForce(double range) {
  d_wallForce.resize(d_particleEnergy.size());
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wallForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  kernelCalcParticleWallForce<<<dimGrid, dimBlock>>>(pRad, pPos, range, wallForce);
  return thrust::reduce(d_wallForce.begin(), d_wallForce.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleWallPressure() {
	 double wallWork = 0, volume = 1;
	 for (long dim = 0; dim < nDim; dim++) {
     volume *= d_boxSize[dim];
	 }
   const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
   const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
   kernelCalcParticleBoxPressure<<<dimGrid, dimBlock>>>(pRad, pPos, wallWork);
	 return wallWork / (nDim * volume);
	 //return totalStress;
}

double SP2D::getParticleDynamicalPressure() {
  double volume = 1;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
  }
  return getParticleTemperature() * numParticles / volume;
}

double SP2D::getParticleTotalPressure() {
  return getParticleVirialPressure() + getParticleDynamicalPressure();
}

double SP2D::getParticleActivePressure(double driving) {
  double activeWork = 0, volume = 1;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
  }
	const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  kernelCalcParticleActivePressure<<<dimGrid, dimBlock>>>(pAngle, pPos, driving, activeWork);

  return activeWork / (nDim * volume);
}

double SP2D::getParticleEnergy() {
  return thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleKineticEnergy() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleTemperature() {
  double ekin = getParticleKineticEnergy();
  return 2 * ekin / (numParticles * nDim);
}

double SP2D::getMassiveTemperature(long firstIndex, double mass) {
  // temperature computed from the massive particles which are set to be the first
  thrust::device_vector<double> velSquared(firstIndex * nDim);
  thrust::transform(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, velSquared.begin(), square());
  return mass * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>()) / (firstIndex * nDim);
}

double SP2D::getParticleDrift() {
  return thrust::reduce(d_particlePos.begin(), d_particlePos.end(), double(0), thrust::plus<double>()) / (numParticles * nDim);
}

//************************* contacts and neighbors ***************************//
thrust::host_vector<long> SP2D::getParticleNeighbors() {
  thrust::host_vector<long> partNeighborListFromDevice;
  partNeighborListFromDevice = d_partNeighborList;
  return partNeighborListFromDevice;
}

void SP2D::calcParticleNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  partMaxNeighbors = thrust::reduce(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncParticleNeighborsToDevice();
  //cout << "SP2D::calcParticleNeighborList: maxNeighbors: " << partMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( partMaxNeighbors > partNeighborListSize ) {
		partNeighborListSize = pow(2, ceil(std::log2(partMaxNeighbors)));
    //cout << "SP2D::calcParticleNeighborList: neighborListSize: " << partNeighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_partNeighborList.resize(numParticles * partNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		syncParticleNeighborsToDevice();
		kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
	}
}

void SP2D::syncParticleNeighborsToDevice() {
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_partNeighborListSize, &partNeighborListSize, sizeof(partNeighborListSize));
	cudaMemcpyToSymbol(d_partMaxNeighbors, &partMaxNeighbors, sizeof(partMaxNeighbors));

	long* partMaxNeighborList = thrust::raw_pointer_cast(&d_partMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_partMaxNeighborListPtr, &partMaxNeighborList, sizeof(partMaxNeighborList));

	long* partNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
	cudaMemcpyToSymbol(d_partNeighborListPtr, &partNeighborList, sizeof(partNeighborList));
}

void SP2D::calcParticleBoxNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleBoxNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  partMaxNeighbors = thrust::reduce(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncParticleNeighborsToDevice();
  //cout << "SP2D::calcParticleNeighborList: maxNeighbors: " << partMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( partMaxNeighbors > partNeighborListSize ) {
		partNeighborListSize = pow(2, ceil(std::log2(partMaxNeighbors)));
    //cout << "SP2D::calcParticleNeighborList: neighborListSize: " << neighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_partNeighborList.resize(numParticles * partNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		syncParticleNeighborsToDevice();
		kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
	}
}

void SP2D::calcParticleContacts(double gapSize) {
  long largestContact = 8*nDim; // Guess
	do {
		//Make a contactList that is the right size
		contactLimit = largestContact;
		d_contactList = thrust::device_vector<long>(numParticles * contactLimit);
		//Prefill the contactList with -1
		thrust::fill(d_contactList.begin(), d_contactList.end(), -1L);
		thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
		//Create device_pointers from thrust arrays
    const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
		long* contactList = thrust::raw_pointer_cast(&d_contactList[0]);
		long* numContacts = thrust::raw_pointer_cast(&d_numContacts[0]);
		kernelCalcParticleContacts<<<dimGrid, dimBlock>>>(pPos, pRad, gapSize, contactLimit, contactList, numContacts);
		//Calculate the maximum number of contacts
		largestContact = thrust::reduce(d_numContacts.begin(), d_numContacts.end(), -1L, thrust::maximum<long>());
    //cout << "SP2D::calcParticleContacts: largestContact = " << largestContact << endl;
	} while(contactLimit < largestContact); // If the guess was not good, do it again
}

//Return normalized contact vectors between every pair of particles in contact
thrust::host_vector<long> SP2D::getContactVectors(double gapSize) {
	//Calculate the set of contacts
	calcParticleContacts(gapSize);
	//Calculate the maximum number of contacts
	maxContacts = thrust::reduce(d_numContacts.begin(), d_numContacts.end(), -1L, thrust::maximum<long>());
	//Create the array to hold the contactVectors
	d_contactVectorList.resize(numParticles * nDim * maxContacts);
	thrust::fill(d_contactVectorList.begin(), d_contactVectorList.end(), double(0));
	double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	long* contactList = thrust::raw_pointer_cast(&d_contactList[0]);
	double* contactVectorList = thrust::raw_pointer_cast(&d_contactVectorList[0]);
	kernelCalcContactVectorList<<<dimGrid, dimBlock>>>(pPos, contactList, d_contactList.size()/numParticles, maxContacts, contactVectorList);
  // convert to host and return
  thrust::host_vector<long> contactVectorListFromDevice;
  contactVectorListFromDevice = d_contactVectorList;
  return contactVectorListFromDevice;
}

//************************** minimizer functions *****************************//
void SP2D::initFIRE(std::vector<double> &FIREparams, long minStep_, long numStep_, long numDOF_) {
  this->fire_ = new FIRE(this);
  if(FIREparams.size() == 7) {
    double a_start_ = FIREparams[0];
    double f_dec_ = FIREparams[1];
    double f_inc_ = FIREparams[2];
    double f_a_ = FIREparams[3];
    double fire_dt_ = FIREparams[4];
    double fire_dt_max_ = FIREparams[5];
    double a_ = FIREparams[6];
    this->fire_->initMinimizer(a_start_, f_dec_, f_inc_, f_a_, fire_dt_, fire_dt_max_, a_, minStep_, numStep_, numDOF_);
  } else {
    cout << "SP2D::initFIRE: wrong number of FIRE parameters, must be 7" << endl;
  }
  resetLastPositions();
}

void SP2D::setParticleMassFIRE() {
  //this->fire_->setParticleMass();
  this->fire_->d_mass.resize(numParticles * nDim);
	for (long particleId = 0; particleId < numParticles; particleId++) {
		for (long dim = 0; dim < nDim; dim++) {
			this->fire_->d_mass[particleId * nDim + dim] = PI / (d_particleRad[particleId] * d_particleRad[particleId]);
		}
	}
}

void SP2D::setTimeStepFIRE(double timeStep_) {
  this->fire_->setFIRETimeStep(timeStep_);
}

void SP2D::particleFIRELoop() {
  this->fire_->minimizerParticleLoop();
}

void SP2D::computeParticleDrift() {
  thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
  double *velSum = thrust::raw_pointer_cast(&d_particleDelta[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  kernelSumParticleVelocity<<<dimGrid, dimBlock>>>(pVel, velSum);
}

void SP2D::conserveParticleMomentum() {
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double *velSum = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelSubtractParticleDrift<<<dimGrid, dimBlock>>>(pVel, velSum);
}

//***************************** NVT integrators ******************************//
void SP2D::initSoftParticleLangevin(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevin2(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinLoop() {
  this->sim_->integrate();
  //computeParticleDrift();
  //conserveParticleMomentum();
  //computeParticleDrift();
  //cout << "velSum: " << thrust::reduce(d_particleVel.begin(), d_particleVel.end(), double(0), thrust::plus<double>()) << endl;
}

void SP2D::initSoftParticleLangevinFixedBox(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevinFixedBox(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleLangevinFixedBox:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinFixedBoxLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleLangevinSubSet(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel) {
  this->sim_ = new SoftParticleLangevinSubSet(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  // subset variables
  this->sim_->firstIndex = firstIndex;
  this->sim_->mass = mass;
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  if(zeroOutMassiveVel == true) {
    thrust::fill(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, double(0));
  }
  cout << "SP2D::initSoftParticleLangevinSubSet:: current temperature: " << setprecision(12) << getParticleTemperature() << " mass: " << this->sim_->mass << endl;
}

void SP2D::softParticleLangevinSubSetLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleLangevinExtField(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevinExtField(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinExtFieldLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleLangevinPerturb(double Temp, double gamma, double extForce, long firstIndex, bool readState) {
  this->sim_ = new SoftParticleLangevinPerturb(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  this->sim_->extForce = extForce;
  this->sim_->firstIndex = firstIndex;
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinPerturbLoop() {
  this->sim_->integrate();
}

//***************************** NVE integrators ******************************//
void SP2D::initSoftParticleNVE(double Temp, bool readState) {
  this->sim_ = new SoftParticleNVE(this, SimConfig(Temp, 0, 0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleNVE:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleNVELoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleNVEFixedBox(double Temp, bool readState) {
  this->sim_ = new SoftParticleNVEFixedBox(this, SimConfig(Temp, 0, 0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleNVEFixedBox:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleNVEFixedBoxLoop() {
  this->sim_->integrate();
}

//**************************** Active integrators ****************************//
void SP2D::initSoftParticleActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveLangevin(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //cout << "SP2D::initSoftParticleActiveLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "SP2D::initSoftParticleActiveLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleActiveLangevinLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleActiveFixedBox(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveFixedBox(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "SP2D::initSoftParticleActiveFixedBox:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleActiveFixedBoxLoop() {
  this->sim_->integrate();

}

void SP2D::initSoftParticleActiveFixedSides(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveFixedSides(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "SP2D::initSoftParticleActiveFixedSides:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleActiveFixedSidesLoop() {
  this->sim_->integrate();

}

void SP2D::initSoftParticleActiveSubSet(double Temp, double Dr, double driving, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel) {
  this->sim_ = new SoftParticleActiveSubSet(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  // subset variables
  this->sim_->firstIndex = firstIndex;
  this->sim_->mass = mass;
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  if(zeroOutMassiveVel == true) {
    thrust::fill(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, double(0));
  }
  cout << "SP2D::initSoftParticleActiveSubSet:: current temperature: " << setprecision(12) << getParticleTemperature() << " mass: " << this->sim_->mass << endl;
}

void SP2D::softParticleActiveSubSetLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleActiveExtField(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveExtField(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "SP2D::initSoftParticleActiveExtField:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleActiveExtFieldLoop() {
  this->sim_->integrate();
}
