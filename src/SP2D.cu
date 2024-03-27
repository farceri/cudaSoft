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
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
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
	simControl.interactionType = simControlStruct::interactionEnum::neighbor;
	simControl.potentialType = simControlStruct::potentialEnum::harmonic;
	simControl.boxType = simControlStruct::boxEnum::harmonic;
	simControl.gravityType = simControlStruct::gravityEnum::off;
	syncSimControlToDevice();
  // default parameters
  dt = 1e-04;
  rho0 = 1;
	ec = 1;
	l1 = 0;
	l2 = 0;
  LEshift = 0;
  gravity = 0;
  ew = 100;
  flowSpeed = 0;
  flowDecay = 1;
  flowViscosity = 1;
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

double SP2D::checkGPUMemory() {
  int device;
  cudaGetDevice(&device);
  //cout << "\nDevice: " << device << endl;
  double free, total, used, mega = 1048576;
  size_t freeInfo,totalInfo;
  cudaMemGetInfo(&freeInfo,&totalInfo);
  free =(uint)freeInfo / mega;
  total =(uint)totalInfo / mega;
  used = total - free;
  //cout << "Memory usage in MB - free: " << freeInfo << " total: " << totalInfo << " used: " << used << endl;
  return used / total;
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
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides2D) {
    cout << "SP2D: setGeometryType: geometryType: fixedSides2D" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides3D) {
    cout << "SP2D: setGeometryType: geometryType: fixedSides3D" << endl;
  } else {
    cout << "SP2D: setGeometryType: please specify valid geometryType: normal, leesEdwards, fixedBox, fixedSides2D or fixedSides3D" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::geometryEnum SP2D::getGeometryType() {
	syncSimControlFromDevice();
	return simControl.geometryType;
}

void SP2D::setInteractionType(simControlStruct::interactionEnum interactionType_) {
	simControl.interactionType = interactionType_;
  if(simControl.interactionType == simControlStruct::interactionEnum::neighbor) {
    cout << "SP2D: setInteractionType: interactionType: neighbor" << endl;
  } else if(simControl.interactionType == simControlStruct::interactionEnum::allToAll) {
    cout << "SP2D: setInteractionType: interactionType: allToAll" << endl;
  } else {
    cout << "SP2D: setInteractionType: please specify valid interactionType: neighbor or allToAll" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::interactionEnum SP2D::getInteractionType() {
	syncSimControlFromDevice();
	return simControl.interactionType;
}

void SP2D::setPotentialType(simControlStruct::potentialEnum potentialType_) {
	simControl.potentialType = potentialType_;
  if(simControl.potentialType == simControlStruct::potentialEnum::harmonic) {
    cout << "SP2D: setPotentialType: potentialType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::lennardJones) {
    cout << "SP2D: setPotentialType: potentialType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::Mie) {
    cout << "SP2D: setPotentialType: potentialType: Mie" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::WCA) {
    cout << "SP2D: setPotentialType: potentialType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::adhesive) {
    cout << "SP2D: setPotentialType: potentialType: adhesive" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::doubleLJ) {
    cout << "SP2D: setPotentialType: potentialType: doubleLJ" << endl;
  } else {
    cout << "SP2D: setPotentialType: please specify valid potentialType: harmonic, lennardJones, WCA, adhesive or doubleLJ" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::potentialEnum SP2D::getPotentialType() {
	syncSimControlFromDevice();
	return simControl.potentialType;
}

void SP2D::setBoxType(simControlStruct::boxEnum boxType_) {
	simControl.boxType = boxType_;
  if(simControl.boxType == simControlStruct::boxEnum::harmonic) {
    cout << "SP2D: setBoxType: boxType: harmonic" << endl;
  } else if(simControl.boxType == simControlStruct::boxEnum::WCA) {
    cout << "SP2D: setBoxType: boxType: WCA" << endl;
  } else {
    cout << "SP2D: setBoxType: please specify valid boxType: on or off" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::boxEnum SP2D::getBoxType() {
	syncSimControlFromDevice();
	return simControl.boxType;
}

void SP2D::setGravityType(simControlStruct::gravityEnum gravityType_) {
	simControl.gravityType = gravityType_;
  if(simControl.gravityType == simControlStruct::gravityEnum::on) {
    cout << "SP2D: setGravityType: gravityType: on" << endl;
  } else if(simControl.gravityType == simControlStruct::gravityEnum::off) {
    cout << "SP2D: setGravityType: gravityType: off" << endl;
  } else {
    cout << "SP2D: setGravityType: please specify valid gravityType: on or off" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::gravityEnum SP2D::getGravityType() {
	syncSimControlFromDevice();
	return simControl.gravityType;
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
		shearPos -= floor(shearPos / boxSize[0]) * boxSize[0];
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
		extendPos -= floor(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double shift_, long direction) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + shift_) * pPos[particleId * d_nDim + direction];
		extendPos -= floor(extendPos / boxSize[direction]) * boxSize[direction];
		pPos[particleId * d_nDim + direction] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyCenteredUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double shift_, long direction) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = pPos[particleId * d_nDim + direction] + shift_ * (pPos[particleId * d_nDim + direction] - boxSize[direction] * 0.5);
		extendPos -= floor(extendPos / boxSize[direction]) * boxSize[direction];
		pPos[particleId * d_nDim + direction] = extendPos;
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
		extendPos -= floor(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
		compressPos = (1 + shiftx_) * pPos[particleId * d_nDim];
		compressPos -= floor(compressPos / boxSize[0]) * boxSize[0];
		pPos[particleId * d_nDim] = compressPos;
	};

	thrust::for_each(r, r+numParticles, biaxialPosition);
}

void SP2D::applyCenteredBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, double shiftx_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto centeredBiaxialPosition = [=] __device__ (long particleId) {
		double extendPos, compressPos;
		extendPos = pPos[particleId * d_nDim + 1] + shifty_ * (pPos[particleId * d_nDim + 1] - boxSize[1] * 0.5);
		extendPos -= floor(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
		compressPos = pPos[particleId * d_nDim] + shiftx_ * (pPos[particleId * d_nDim] - boxSize[0] * 0.5);
		compressPos -= floor(compressPos / boxSize[0]) * boxSize[0];
		pPos[particleId * d_nDim] = compressPos;
	};

	thrust::for_each(r, r+numParticles, centeredBiaxialPosition);
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

double SP2D::getMaxParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(-1), thrust::maximum<double>());
}

void SP2D::setParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
}

void SP2D::setPBC() {
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  kernelCheckParticlePBC<<<dimGrid, dimBlock>>>(pPosPBC, pPos);
  // copy to device
  d_particlePos = d_particlePosPBC;
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

void SP2D::resetLastVelocities() {
  d_particleLastVel = d_particleVel;
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
  if(nDim == 2) {
    thrust::device_vector<double> d_radSquared(numParticles);
    thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radSquared.begin(), square());
    return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / (d_boxSize[0] * d_boxSize[1]);
  } else if(nDim == 3) {
    thrust::device_vector<double> d_radCubed(numParticles);
    thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radCubed.begin(), cube());
    return thrust::reduce(d_radCubed.begin(), d_radCubed.end(), double(0), thrust::plus<double>()) * 3 * PI / (4 * d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
  } else {
    cout << "SP2D::getParticlePhi: only dimensions 2 and 3 are allowed!" << endl;
    return 0;
  }
}

double SP2D::getParticleMSD() {
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelCalcParticleDistanceSq<<<dimGrid,dimBlock>>>(particlePos, particleInitPos, particleDelta);
  return thrust::reduce(d_particleDelta.begin(), d_particleDelta.end(), double(0), thrust::plus<double>()) / numParticles;
}

double SP2D::setDisplacementCutoff(double cutoff_) {
  switch (simControl.potentialType) {
    case simControlStruct::potentialEnum::harmonic:
    cutDistance = 1;
    break;
    case simControlStruct::potentialEnum::lennardJones:
    cutDistance = LJcutoff;
    break;
    case simControlStruct::potentialEnum::WCA:
    cutDistance = WCAcut;
    break;
    case simControlStruct::potentialEnum::Mie:
    cutDistance = LJcutoff;
    break;
    case simControlStruct::potentialEnum::adhesive:
    cutDistance = l2;
    break;
    case simControlStruct::potentialEnum::doubleLJ:
    cutDistance = LJcutoff;
    break;
  }
  cutDistance += cutoff_;
  cutoff = cutoff_ * 2 * getMeanParticleSigma();
  cout << "DPM2D::setDisplacementCutoff - cutDistance: " << cutDistance << " cutoff: " << cutoff << endl;
  return cutDistance;
}

double SP2D::getParticleMaxDisplacement() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, pDisp);
  return thrust::reduce(d_particleDisp.begin(), d_particleDisp.end(), double(-1), thrust::maximum<double>());
}

void SP2D::checkParticleMaxDisplacement() {
  double maxDelta;
  maxDelta = getParticleMaxDisplacement();
  if(2 * maxDelta > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta1: " << maxDelta1 << " cutoff: " << cutoff << endl;
  }
}

void SP2D::checkParticleMaxDisplacement2() {
  double maxDelta1, maxDelta2;
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, pDisp);
  thrust::sort(d_particleDisp.begin(), d_particleDisp.end(), thrust::greater<double>());
  thrust::host_vector<double> sorted_Disp = d_particleDisp;
  maxDelta1 = sorted_Disp[0];
  maxDelta2 = sorted_Disp[1];
  if(3 * maxDelta1 > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta1: " << maxDelta1 << " cutoff: " << cutoff << endl;
  } else if(3 * maxDelta2 > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta2: " << maxDelta2 << " cutoff: " << cutoff << endl;
  }
}

void SP2D::resetUpdateCount() {
  updateCount = double(0);
  //cout << "SP2D::resetUpdateCount - updatCount " << updateCount << endl;
}

long SP2D::getUpdateCount() {
  return updateCount;
}

double SP2D::getSoftWaveNumber() {
  if(nDim == 2) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getParticlePhi() / (PI * numParticles)));
  } else if (nDim == 3) {
    return PI / (2. * cbrt(d_boxSize[0] * d_boxSize[1] * d_boxSize[2] * getParticlePhi() / (PI * numParticles)));
  } else {
    cout << "SP2D::getSoftWaveNumber: only dimensions 2 and 3 are allowed!" << endl;
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
void SP2D::setPolyRandomParticles(double phi0, double polyDispersity) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale, boxLength = 1.;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = 0.5 * exp(mean + randNum * sigma);
  }
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    scale = cbrt(getParticlePhi() / phi0);
  } else {
    cout << "SP2D::setScaledPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
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

void SP2D::setScaledPolyRandomParticles(double phi0, double polyDispersity, double lx) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = 0.5 * exp(mean + randNum * sigma);
  }
  boxSize[0] = lx;
  for (long dim = 1; dim < nDim; dim++) {
    boxSize[dim] = 1;
  }
  setBoxSize(boxSize);
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    scale = cbrt(getParticlePhi() / phi0);
  } else {
    cout << "SP2D::setScaledPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
  boxSize[0] = lx * scale;
  for (long dim = 1; dim < nDim; dim++) {
    boxSize[dim] = scale;
  }
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

void SP2D::setScaledMonoRandomParticles(double phi0, double lx) {
  thrust::host_vector<double> boxSize(nDim);
  double scale;
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] = 0.5;
  }
  boxSize[0] = lx;
  for (long dim = 1; dim < nDim; dim++) {
    boxSize[dim] = 1;
  }
  setBoxSize(boxSize);
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    scale = cbrt(getParticlePhi() / phi0);
  } else {
    cout << "SP2D::setScaledPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
  boxSize[0] = lx * scale;
  for (long dim = 1; dim < nDim; dim++) {
    boxSize[dim] = scale;
  }
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

void SP2D::pressureScaleParticles(double pscale) {
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(pscale), d_particlePos.begin(), thrust::multiplies<double>());
  thrust::transform(d_boxSize.begin(), d_boxSize.end(), thrust::make_constant_iterator(pscale), d_boxSize.begin(), thrust::multiplies<double>());
}

void SP2D::scaleParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
}

void SP2D::scaleParticlePacking() {
  double sigma = 2 * getMeanParticleSigma();
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

double SP2D::setTimeStep(double dt_) {
  dt = dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

void SP2D::setAdhesionParams(double l1_, double l2_) {
  l1 = l1_;
  l2 = l2_;
  cudaMemcpyToSymbol(d_l1, &l1, sizeof(l1));
  cudaMemcpyToSymbol(d_l2, &l2, sizeof(l2));
}

void SP2D::setLJcutoff(double LJcutoff_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  LJecut = 4 * ec * (1 / pow(LJcutoff, 12) - 1 / pow(LJcutoff, 6));
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
  cout << "SP2D::setLJcutoff: LJcutoff: " << LJcutoff << " LJecut: " << LJecut << endl;
}

void SP2D::setDoubleLJconstants(double LJcutoff_, double eAA_, double eAB_, double eBB_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  eAA = eAA_;
  eAB = eAB_;
  eBB = eBB_;
  cudaMemcpyToSymbol(d_eAA, &eAA, sizeof(eAA));
  cudaMemcpyToSymbol(d_eAB, &eAB, sizeof(eAB));
  cudaMemcpyToSymbol(d_eBB, &eBB, sizeof(eBB));
  LJecut = 4 * (1 / pow(LJcutoff, 12) - 1 / pow(LJcutoff, 6));
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
  cout << "SP2D::setDoubleLJconstants: eAA: " << eAA << " eAB: " << eAB << " eBB: " << eBB << " LJcutoff: " << LJcutoff << " LJecut: " << LJecut << endl;
}

void SP2D::setMieParams(double LJcutoff_, double nPower_, double mPower_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  nPower = nPower_;
  mPower = mPower_;
  cudaMemcpyToSymbol(d_nPower, &nPower, sizeof(nPower));
  cudaMemcpyToSymbol(d_mPower, &mPower, sizeof(mPower));
  double pRatio = nPower / mPower;
  double pDiff = nPower - mPower;
  double mieConstant = (nPower / pDiff) * pow(pRatio, mPower / pDiff);
  cudaMemcpyToSymbol(d_mieConstant, &mieConstant, sizeof(mieConstant));
  double Miecut = mieConstant * ec * (1 / pow(LJcutoff, nPower) - 1 / pow(LJcutoff, mPower));
  cudaMemcpyToSymbol(d_Miecut, &Miecut, sizeof(Miecut));
  cout << "SP2D::setMieParams: LJcutoff: " << LJcutoff << " Miecut: " << Miecut << " n: " << nPower << " m: " << mPower << endl;
}

void SP2D::setBoxEnergyScale(double ew_) {
  ew = ew_;
  cudaMemcpyToSymbol(d_ew, &ew, sizeof(ew));
}

void SP2D::setGravity(double gravity_, double ew_) {
  gravity = gravity_;
  ew = ew_;
  cudaMemcpyToSymbol(d_gravity, &gravity, sizeof(gravity));
  cudaMemcpyToSymbol(d_ew, &ew, sizeof(ew));
}

void SP2D::setFluidFlow(double speed_, double viscosity_) {
  flowSpeed = speed_;
  flowViscosity = viscosity_;
  cudaMemcpyToSymbol(d_flowSpeed, &flowSpeed, sizeof(flowSpeed));
  cudaMemcpyToSymbol(d_flowViscosity, &flowViscosity, sizeof(flowViscosity));
  d_surfaceHeight.resize(numParticles);
  thrust::fill(d_surfaceHeight.begin(), d_surfaceHeight.end(), double(0));
  d_flowVel.resize(numParticles * nDim);
  thrust::fill(d_flowVel.begin(), d_flowVel.end(), double(0));
}

void SP2D::calcSurfaceHeight() {
  flowHeight = 0;
  calcParticleContacts(0);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const long *numContacts = thrust::raw_pointer_cast(&d_numContacts[0]);
	double *sHeight = thrust::raw_pointer_cast(&d_surfaceHeight[0]);
  kernelCalcSurfaceHeight<<<dimGrid, dimBlock>>>(pPos, numContacts, sHeight);
  flowHeight = 0.95 * thrust::reduce(d_surfaceHeight.begin(), d_surfaceHeight.end(), double(-1), thrust::maximum<double>());
  thrust::fill(d_surfaceHeight.begin(), d_surfaceHeight.end(), flowHeight);
  // set flowDecay proportional to flowHeight
  flowDecay = flowHeight / 2;
  cudaMemcpyToSymbol(d_flowDecay, &flowDecay, sizeof(flowDecay));
}

double SP2D::getSurfaceHeight() {
  return flowHeight;
}

void SP2D::calcFlowVelocity() {
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *sHeight = thrust::raw_pointer_cast(&d_surfaceHeight[0]);
	double *flowVel = thrust::raw_pointer_cast(&d_flowVel[0]);
  kernelCalcFlowVelocity<<<dimGrid, dimBlock>>>(pPos, sHeight, flowVel);
}

/*void SP2D::calcParticleForceEnergy() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}*/

void SP2D::calcParticleForceEnergy() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::neighbor:
    kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
    case simControlStruct::interactionEnum::allToAll:
    kernelCalcAllToAllParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
  }
  switch (simControl.geometryType) {
		case simControlStruct::geometryEnum::fixedBox:
    kernelCalcParticleBoxInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
    kernelCalcParticleSidesInteraction2D<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
    case simControlStruct::geometryEnum::fixedSides3D:
    kernelCalcParticleSidesInteraction3D<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
	}
  switch (simControl.gravityType) {
    case simControlStruct::gravityEnum::on:
    kernelAddParticleGravity<<<dimGrid, dimBlock>>>(pPos, pForce, pEnergy);
    break;
  }
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
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *pStress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcParticleStressTensor<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, pStress);
}

double SP2D::getParticlePressure() {
  calcParticleStressTensor();
  if(nDim == 2) {
    return (d_stress[0] + d_stress[3]) / (nDim * d_boxSize[0] * d_boxSize[1]);
  } else {
    return 0;
  }
}

double SP2D::getParticleSurfaceTension() {
  calcParticleStressTensor();
  if(nDim == 2) {
    return 0.5 * d_boxSize[0] * (d_stress[0] - d_stress[3]) / (d_boxSize[0] * d_boxSize[1]);
  } else {
    return 0;
  }
}

double SP2D::getParticleShearStress() {
  calcParticleStressTensor();
  if(nDim == 2) {
    return (d_stress[1] + d_stress[2]) / (nDim * d_boxSize[0] * d_boxSize[1]);
  } else {
    return 0;
  }
}

double SP2D::getParticleExtensileStress() {
   calcParticleStressTensor();
	 return d_stress[3] / (d_boxSize[0] * d_boxSize[1]);
}

double SP2D::getParticleWallForce(double range) {
  // first get pbc positions
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  d_particlePosPBC = getPBCParticlePositions();
  // then use them to compute the force across the wall
  d_wallForce.resize(d_particleEnergy.size());
  d_wallCount.resize(d_particleEnergy.size());
  thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
  thrust::fill(d_wallCount.begin(), d_wallCount.end(), long(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  double *wallForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  long *wallCount = thrust::raw_pointer_cast(&d_wallCount[0]);
  kernelCalcParticleWallForce<<<dimGrid, dimBlock>>>(pRad, pPosPBC, range, wallForce, wallCount);
  return thrust::reduce(d_wallForce.begin(), d_wallForce.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleActiveWallForce(double range, double driving) {
  // first get pbc positions
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  d_particlePosPBC = getPBCParticlePositions();
  // then use them to compute the force across the wall
  d_wallForce.resize(d_particleEnergy.size());
  d_wallCount.resize(d_particleEnergy.size());
  thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
  thrust::fill(d_wallCount.begin(), d_wallCount.end(), long(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  double *wallForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  long *wallCount = thrust::raw_pointer_cast(&d_wallCount[0]);
  kernelCalcParticleWallForce<<<dimGrid, dimBlock>>>(pRad, pPosPBC, range, wallForce, wallCount);
  const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  kernelAddParticleWallActiveForce<<<dimGrid, dimBlock>>>(pAngle, driving, wallForce, wallCount);
  return thrust::reduce(d_wallForce.begin(), d_wallForce.end(), double(0), thrust::plus<double>());
}

long SP2D::getTotalParticleWallCount() {
  return thrust::reduce(d_wallCount.begin(), d_wallCount.end(), long(0), thrust::plus<long>());
}

double SP2D::getParticleWallPressure() {
	 double volume = 1;
	 for (long dim = 0; dim < nDim; dim++) {
     volume *= d_boxSize[dim];
	 }
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wallStress = thrust::raw_pointer_cast(&d_stress[0]);
  switch (simControl.geometryType) {
		case simControlStruct::geometryEnum::fixedBox:
    kernelCalcParticleBoxStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
    kernelCalcParticleSides2DStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
    break;
	}
	return (d_stress[0] + d_stress[3]) / (nDim * volume);
	//return totalStress;
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

double SP2D::getParticlePotentialEnergy() {
  return thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleKineticEnergy() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  return thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleEnergy() {
  double etot = getParticlePotentialEnergy();
  etot = etot + getParticleKineticEnergy();
  return etot;
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
  cout << "SP2D::initSoftParticleLangevinExtField:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
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
  cout << "SP2D::initSoftParticleLangevinPerturb:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinPerturbLoop() {
  this->sim_->integrate();
}

//********************** Langevin fluid flow integrator *********************//
void SP2D::initSoftParticleLangevinFlow(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevinFlow(this, SimConfig(Temp, 0, 0));
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
  calcSurfaceHeight();
  calcFlowVelocity();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleLangevinFlow:: current temperature: " << setprecision(10) << getParticleTemperature() << " surface height: " << getSurfaceHeight() << endl;
}

void SP2D::softParticleLangevinFlowLoop() {
  this->sim_->integrate();
}

//************************** Fluid flow integrator ***************************//
void SP2D::initSoftParticleFlow(double gamma, bool readState) {
  this->sim_ = new SoftParticleFlow(this, SimConfig(0, 0, 0));
  this->sim_->gamma = gamma;
  resetLastPositions();
  calcSurfaceHeight();
  calcFlowVelocity();
  cout << "SP2D::initSoftParticleFlow:: current temperature: " << setprecision(10) << getParticleTemperature() << " surface height: " << getSurfaceHeight() << endl;
}

void SP2D::softParticleFlowLoop() {
  this->sim_->integrate();
}


//***************************** NVE integrator *******************************//
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
