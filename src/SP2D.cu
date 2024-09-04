//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// FUNCTION DECLARATIONS

#include "../include/SP2D.h"
#include "../include/defs.h"
#include "../include/cudaKernel.cuh"
#include "../include/Simulator.h"
#include "../include/FIRE.h"
#include "../include/cached_allocator.cuh"
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
#include <thrust/gather.h>
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
  num1 = numParticles;
  // the default is monodisperse size distribution
  setDimBlock(dimBlock);
  setNDim(nDim);
  setNumParticles(numParticles);
	simControl.particleType = simControlStruct::particleEnum::passive;
	simControl.geometryType = simControlStruct::geometryEnum::normal;
	simControl.neighborType = simControlStruct::neighborEnum::neighbor;
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
  shift = false;
  d_boxSize.resize(nDim);
  thrust::fill(d_boxSize.begin(), d_boxSize.end(), double(1));
  d_stress.resize(nDim * nDim);
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  // particle variables
  initParticleVariables(numParticles);
  initParticleDeltaVariables(numParticles);
  // initialize contacts and neighbors
  //initContacts(numParticles);
  initParticleNeighbors(numParticles);
  syncParticleNeighborsToDevice();
  if(cudaGetLastError()) cout << "SP2D():: cudaGetLastError(): " << cudaGetLastError() << endl;
}

SP2D::~SP2D() {}

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
  if(nDim == 2) {
    d_particleAngle.resize(numParticles_);
  } else if(nDim == 3) {
    d_particleAngle.resize(numParticles_ * nDim);
  }
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

void SP2D::setParticleType(simControlStruct::particleEnum particleType_) {
	simControl.particleType = particleType_;
  if(simControl.particleType == simControlStruct::particleEnum::passive) {
    cout << "SP2D::setParticleType: particleType: passive" << endl;
  } else if(simControl.particleType == simControlStruct::particleEnum::active) {
    if(nDim == 2) {
      d_activeAngle.resize(numParticles);
    } else if(nDim == 3) {
      d_activeAngle.resize(numParticles * nDim);
    }
    thrust::fill(d_activeAngle.begin(), d_activeAngle.end(), double(0));
    cout << "SP2D::setParticleType: particleType: active" << endl;
  } else {
    cout << "SP2D::setParticleType: please specify valid particleType: passive or active" << endl;
  }
	syncSimControlToDevice();
}

void SP2D::setGeometryType(simControlStruct::geometryEnum geometryType_) {
	simControl.geometryType = geometryType_;
  if(simControl.geometryType == simControlStruct::geometryEnum::normal) {
    cout << "SP2D::setGeometryType: geometryType: normal" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::leesEdwards) {
    cout << "SP2D::setGeometryType: geometryType: leesEdwards" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedBox) {
    cout << "SP2D::setGeometryType: geometryType: fixedBox" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides2D) {
    cout << "SP2D:;setGeometryType: geometryType: fixedSides2D" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides3D) {
    cout << "SP2D::setGeometryType: geometryType: fixedSides3D" << endl;
  } else {
    cout << "SP2D::setGeometryType: please specify valid geometryType: normal, leesEdwards, fixedBox, fixedSides2D or fixedSides3D" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::geometryEnum SP2D::getGeometryType() {
	syncSimControlFromDevice();
	return simControl.geometryType;
}

void SP2D::setNeighborType(simControlStruct::neighborEnum neighborType_) {
	simControl.neighborType = neighborType_;
  if(simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
    cout << "SP2D::setNeighborType: neighborType: neighbor" << endl;
  } else if(simControl.neighborType == simControlStruct::neighborEnum::allToAll) {
    cout << "SP2D::setNeighborType: neighborType: allToAll" << endl;
  } else {
    cout << "SP2D::setNeighborType: please specify valid neighborType: neighbor or allToAll" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::neighborEnum SP2D::getNeighborType() {
	syncSimControlFromDevice();
	return simControl.neighborType;
}

void SP2D::setPotentialType(simControlStruct::potentialEnum potentialType_) {
	simControl.potentialType = potentialType_;
  if(simControl.potentialType == simControlStruct::potentialEnum::harmonic) {
    cout << "SP2D::setPotentialType: potentialType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::lennardJones) {
    cout << "SP2D::setPotentialType: potentialType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::Mie) {
    cout << "SP2D::setPotentialType: potentialType: Mie" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::WCA) {
    cout << "SP2D::setPotentialType: potentialType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::adhesive) {
    cout << "SP2D::setPotentialType: potentialType: adhesive" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::doubleLJ) {
    cout << "SP2D::setPotentialType: potentialType: doubleLJ" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::LJMinusPlus) {
    cout << "SP2D::setPotentialType: potentialType: LJMinusPlus" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::LJWCA) {
    cout << "SP2D::setPotentialType: potentialType: LJWCA" << endl;
  } else {
    cout << "SP2D::setPotentialType: please specify valid potentialType: harmonic, lennardJones, WCA, adhesive, doubleLJ, LJMinusPlus and LJWCA" << endl;
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
    cout << "SP2D::setBoxType: boxType: harmonic" << endl;
  } else if(simControl.boxType == simControlStruct::boxEnum::WCA) {
    cout << "SP2D::setBoxType: boxType: WCA" << endl;
  } else {
    cout << "SP2D::setBoxType: please specify valid boxType: on or off" << endl;
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
    cout << "SP2D::setGravityType: gravityType: on" << endl;
  } else if(simControl.gravityType == simControlStruct::gravityEnum::off) {
    cout << "SP2D::setGravityType: gravityType: off" << endl;
  } else {
    cout << "SP2D::setGravityType: please specify valid gravityType: on or off" << endl;
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

void SP2D::applyExtension(double strainy_) {
  // first set the new boxSize
  thrust::host_vector<double> newBoxSize(nDim);
  newBoxSize = getBoxSize();
  newBoxSize[1] = (1 + strainy_) * newBoxSize[1];
  setBoxSize(newBoxSize);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + strainy_) * pPos[particleId * d_nDim + 1];
		extendPos -= floor(extendPos / boxSize[1]) * boxSize[1];
		pPos[particleId * d_nDim + 1] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = (1 + strain_) * pPos[particleId * d_nDim + direction_];
		extendPos -= floor(extendPos / boxSize[direction_]) * boxSize[direction_];
		pPos[particleId * d_nDim + direction_] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyCenteredUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto extendPosition = [=] __device__ (long particleId) {
		double extendPos;
		extendPos = pPos[particleId * d_nDim + direction_] + strain_ * (pPos[particleId * d_nDim + direction_] - boxSize[direction_] * 0.5);
		extendPos -= floor(extendPos / boxSize[direction_]) * boxSize[direction_];
		pPos[particleId * d_nDim + direction_] = extendPos;
	};

	thrust::for_each(r, r+numParticles, extendPosition);
}

void SP2D::applyBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

  double otherStrain = -strain_ / (1 + strain_);

	auto biaxialPosition = [=] __device__ (long particleId) {
		double extendPos, compressPos;
		extendPos = (1 + strain_) * pPos[particleId * d_nDim + direction_];
		extendPos -= floor(extendPos / boxSize[direction_]) * boxSize[direction_];
		pPos[particleId * d_nDim + direction_] = extendPos;
		compressPos = (1 + otherStrain) * pPos[particleId * d_nDim + !direction_];
		compressPos -= floor(compressPos / boxSize[!direction_]) * boxSize[!direction_];
		pPos[particleId * d_nDim + !direction_] = compressPos;
	};

	thrust::for_each(r, r+numParticles, biaxialPosition);
}

void SP2D::applyBiaxialExpExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

	auto biaxialExpPosition = [=] __device__ (long particleId) {
		double extendPos, compressPos;
		extendPos = exp(strain_) * pPos[particleId * d_nDim + direction_];
		extendPos -= floor(extendPos / boxSize[direction_]) * boxSize[direction_];
		pPos[particleId * d_nDim + direction_] = extendPos;
		compressPos = exp(-strain_) * pPos[particleId * d_nDim + !direction_];
		compressPos -= floor(compressPos / boxSize[!direction_]) * boxSize[!direction_];
		pPos[particleId * d_nDim + !direction_] = compressPos;
	};

	thrust::for_each(r, r+numParticles, biaxialExpPosition);
}

void SP2D::applyCenteredBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newBoxSize_);
	auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

  double otherStrain = -strain_ / (1 + strain_);

	auto centeredBiaxialPosition = [=] __device__ (long particleId) {
		double extendPos, compressPos;
		extendPos = pPos[particleId * d_nDim + direction_] + strain_ * (pPos[particleId * d_nDim + direction_] - boxSize[direction_] * 0.5);
		extendPos -= floor(extendPos / boxSize[direction_]) * boxSize[direction_];
		pPos[particleId * d_nDim + direction_] = extendPos;
		compressPos = pPos[particleId * d_nDim + !direction_] + otherStrain * (pPos[particleId * d_nDim + !direction_] - boxSize[!direction_] * 0.5);
		compressPos -= floor(compressPos / boxSize[!direction_]) * boxSize[!direction_];
		pPos[particleId * d_nDim + !direction_] = compressPos;
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

long SP2D::getTypeNumParticles() {
  long num1FromDevice;
  cudaMemcpyFromSymbol(&num1FromDevice, d_num1, sizeof(d_num1));
	return num1FromDevice;
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
  cudaDeviceSynchronize();
  d_particleLastPos = d_particlePos;
}

void SP2D::setInitialPositions() {
  cudaDeviceSynchronize();
  d_particleInitPos = d_particlePos;
}

thrust::host_vector<double> SP2D::getLastPositions() {
  thrust::host_vector<double> lastPosFromDevice;
  lastPosFromDevice = d_particleLastPos;
  return lastPosFromDevice;
}

void SP2D::resetLastVelocities() {
  cudaDeviceSynchronize();
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
    return thrust::reduce(d_radCubed.begin(), d_radCubed.end(), double(0), thrust::plus<double>()) * 4 * PI / (3 * d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
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
    cutDistance = 1 + l2;
    break;
    case simControlStruct::potentialEnum::doubleLJ:
    cutDistance = LJcutoff;
    break;
    case simControlStruct::potentialEnum::LJMinusPlus:
    cutDistance = LJcutoff;
    break;
    case simControlStruct::potentialEnum::LJWCA:
    cutDistance = LJcutoff;
    break;
    default:
    break;
  }
  cutDistance += cutoff_; // adimensional because it is used for the overlap (gap) between two particles
  cutoff = cutoff_ * 2 * getMeanParticleSigma();
  cout << "SP2D::setDisplacementCutoff - cutDistance: " << cutDistance << " cutoff: " << cutoff << endl;
  return cutDistance;
}

// this function is called after particleDisplacement has been computed
void SP2D::removeCOMDrift() {
  // compute drift on x
  thrust::device_vector<double> particleDisp_x(d_particleDisp.size() / 2);
  thrust::device_vector<long> idx(d_particleDisp.size() / 2);
  thrust::sequence(idx.begin(), idx.end(), 0, 2);
  thrust::gather(idx.begin(), idx.end(), d_particleDisp.begin(), particleDisp_x.begin());
  double drift_x = thrust::reduce(particleDisp_x.begin(), particleDisp_x.end(), double(0), thrust::plus<double>()) / numParticles;
  // compute drift on y
  thrust::device_vector<double> particleDisp_y(d_particleDisp.size() / 2);
  thrust::device_vector<long> idy(d_particleDisp.size() / 2);
  thrust::sequence(idy.begin(), idy.end(), 1, 2);
  thrust::gather(idy.begin(), idy.end(), d_particleDisp.begin(), particleDisp_y.begin());
  double drift_y = thrust::reduce(particleDisp_y.begin(), particleDisp_y.end(), double(0), thrust::plus<double>()) / numParticles;

  // subtract drift from current positions
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double* boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

  auto removeDrift = [=] __device__ (long pId) {
		pPos[pId * s_nDim] -= drift_x;
    pPos[pId * s_nDim + 1] -= drift_y;
  };
  thrust::for_each(r, r + numParticles, removeDrift);
}

double SP2D::getParticleMaxDisplacement() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, pDisp);
  return thrust::reduce(d_particleDisp.begin(), d_particleDisp.end(), double(-1), thrust::maximum<double>());
}

void SP2D::checkParticleDisplacement() {
  //const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  thrust::device_vector<int> recalcFlag(d_particleRad.size());
  thrust::fill(recalcFlag.begin(), recalcFlag.end(), int(0));
  int *flag = thrust::raw_pointer_cast(&recalcFlag[0]);
  kernelCheckParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, flag, cutoff);
  int sumFlag = thrust::reduce(recalcFlag.begin(), recalcFlag.end(), int(0), thrust::plus<int>());
  if(sumFlag != 0) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    if(shift == true) {
      removeCOMDrift();
    }
    updateCount += 1;
  }
}

void SP2D::resetUpdateCount() {
  updateCount = double(0);
  //cout << "SP2D::resetUpdateCount - updatCount " << updateCount << endl;
}

long SP2D::getUpdateCount() {
  return updateCount;
}

void SP2D::checkParticleMaxDisplacement() {
  double maxDelta = getParticleMaxDisplacement();
  if(3 * maxDelta > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    if(shift == true) {
      removeCOMDrift();
    }
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta: " << maxDelta << " cutoff: " << cutoff << endl;
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
  double maxSum = maxDelta1 + maxDelta2;
  if(3 * maxSum > cutoff) {
    calcParticleNeighborList(cutDistance);
    resetLastPositions();
    if(shift == true) {
      removeCOMDrift();
    }
    updateCount += 1;
    //cout << "SP2D::checkParticleMaxDisplacement - updated neighbors, maxDelta2 + maxDelta1: " << maxSum << " " << maxDelta1 << " " << maxDelta2 << " cutoff: " << cutoff << endl;
  }
}

void SP2D::checkParticleNeighbors() {
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    checkParticleDisplacement();
    //checkParticleMaxDisplacement();
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
    default:
    break;
  }
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
    cout << "SP2D::setPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
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

void SP2D::setScaledPolyRandomParticles(double phi0, double polyDispersity, double lx, double ly, double lz) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean = 0, sigma, scale;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = 0.5 * exp(mean + randNum * sigma);
  }
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    boxSize[2] = lz;
    setBoxSize(boxSize);
    scale = cbrt(getParticlePhi() / phi0);
    boxSize[2] = lz * scale;
  } else {
    cout << "SP2D::setScaledPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
  boxSize[0] = lx * scale;
  boxSize[1] = ly * scale;
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

void SP2D::setScaledMonoRandomParticles(double phi0, double lx, double ly, double lz) {
  thrust::host_vector<double> boxSize(nDim);
  double scale;
  // generate polydisperse particle size
  thrust::fill(d_particleRad.begin(), d_particleRad.end(), 0.5);
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    boxSize[2] = lz;
    setBoxSize(boxSize);
    scale = cbrt(getParticlePhi() / phi0);
    boxSize[2] = lz * scale;
  } else {
    cout << "SP2D::setScaledPolyRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
  boxSize[0] = lx * scale;
  boxSize[1] = ly * scale;
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

void SP2D::setScaledBiRandomParticles(double phi0, double lx, double ly, double lz) {
  thrust::host_vector<double> boxSize(nDim);
  double scale;
  long halfNum = int(numParticles / 2);
  // generate polydisperse particle size
  thrust::fill(d_particleRad.begin(), d_particleRad.begin() + halfNum, 0.5);
  thrust::fill(d_particleRad.begin() + halfNum, d_particleRad.end(), 0.7);
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  if(nDim == 2) {
    scale = sqrt(getParticlePhi() / phi0);
  } else if(nDim == 3) {
    boxSize[2] = lz;
    setBoxSize(boxSize);
    scale = cbrt(getParticlePhi() / phi0);
    boxSize[2] = lz * scale;
  } else {
    cout << "SP2D::setScaledBiRandomSoftParticles: only dimesions 2 and 3 are allowed!" << endl;
  }
  boxSize[0] = lx * scale;
  boxSize[1] = ly * scale;
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

  if(nDim == 2) {
    auto computeParticleAngle2D = [=] __device__ (long particleId) {
      pAngle[particleId] = atan(pVel[particleId * p_nDim + 1] / pVel[particleId * p_nDim]);
    };

    thrust::for_each(r, r + numParticles, computeParticleAngle2D);
    
  } else if(nDim == 3) {
      auto computeParticleAngle3D = [=] __device__ (long particleId) {
      auto theta = acos(pVel[particleId * p_nDim + 2]);
      auto phi = atan(pVel[particleId * p_nDim + 1] / pVel[particleId * p_nDim]);
      pAngle[particleId * p_nDim] = cos(theta) * cos(phi);
      pAngle[particleId * p_nDim + 1] = sin(theta) * cos(phi);
      pAngle[particleId * p_nDim + 2] = sin(phi);
    };

    thrust::for_each(r, r + numParticles, computeParticleAngle3D);
  }
}

//*************************** force and energy *******************************//
void SP2D::setEnergyCostant(double ec_) {
  ec = ec_;
  cudaMemcpyToSymbol(d_ec, &ec, sizeof(ec));
  setBoxEnergyScale(ec);
}

double SP2D::getEnergyCostant() {
  if(simControl.potentialType == simControlStruct::potentialEnum::doubleLJ) {
    return (eAA + eBB) * 0.5;
  } else {
    return ec;
  }
}

double SP2D::setTimeStep(double dt_) {
  dt = dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

void SP2D::setSelfPropulsionParams(double driving_, double taup_) {
  driving = driving_;
  taup = taup_;
  cudaMemcpyToSymbol(d_driving, &driving, sizeof(driving));
  cudaMemcpyToSymbol(d_taup, &taup, sizeof(taup));
  //cout << "SP2D::setSelfPropulsionParams:: driving: " << driving << " taup: " << taup << endl;
}

void SP2D::getSelfPropulsionParams(double &driving_, double &taup_) {
  driving_ = driving;
  taup_ = taup;
  //cout << "SP2D::getSelfPropulsionParams:: driving: " << driving_ << " taup: " << taup_ << endl;
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
  double ratio6 = 1 / pow(LJcutoff, 6);
  LJecut = 4 * ec * (ratio6 * ratio6 - ratio6);
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
	LJfshift = 24 * ec * (2 * ratio6 - 1) * ratio6 / LJcutoff;
  cudaMemcpyToSymbol(d_LJfshift, &LJfshift, sizeof(LJfshift));
  cout << "SP2D::setLJcutoff::LJcutoff: " << LJcutoff << " energy shift: " << LJecut << " LJfshift: " << LJfshift << endl;
}

void SP2D::setDoubleLJconstants(double LJcutoff_, double eAA_, double eAB_, double eBB_, long num1_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  eAA = eAA_;
  eAB = eAB_;
  eBB = eBB_;
  cudaMemcpyToSymbol(d_eAA, &eAA, sizeof(eAA));
  cudaMemcpyToSymbol(d_eAB, &eAB, sizeof(eAB));
  cudaMemcpyToSymbol(d_eBB, &eBB, sizeof(eBB));
  setBoxEnergyScale(eAA);
  double ratio6 = 1 / pow(LJcutoff, 6);
  LJecut = 4 * (ratio6 * ratio6 - ratio6);
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
	LJfshift = 24 * (2 * ratio6 - 1) * ratio6 / LJcutoff;
  cudaMemcpyToSymbol(d_LJfshift, &LJfshift, sizeof(LJfshift));
  num1 = num1_;
  cudaMemcpyToSymbol(d_num1, &num1, sizeof(num1));
  long num1FromDevice = 0;
  cudaMemcpyFromSymbol(&num1FromDevice, d_num1, sizeof(d_num1));
  cout << "SP2D::setDoubleLJconstants::eAA: " << eAA << " eAB: " << eAB << " eBB: " << eBB;
  cout << " LJcutoff: " << LJcutoff << " LJecut: " << LJecut << " LJfshift: " << LJfshift;
  cout << " num1: " << num1 << " from device: " << num1FromDevice << endl;
}

void SP2D::setLJWCAparams(double LJcutoff_, long num1_) {
  setLJcutoff(LJcutoff_);
  num1 = num1_;
  cudaMemcpyToSymbol(d_num1, &num1, sizeof(num1));
  long num1FromDevice = 0;
  cudaMemcpyFromSymbol(&num1FromDevice, d_num1, sizeof(d_num1));
  cout << "SP2D::setLJWCAparams::num1: " << num1 << " from device: " << num1FromDevice << endl;
}

void SP2D::setLJMinusPlusParams(double LJcutoff_, long num1_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  double ratio6 = 1 / pow(LJcutoff, 6);
  LJecut = 4 * ec * (ratio6 * ratio6 - ratio6);
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
	LJfshift = 24 * ec * (2 * ratio6 - 1) * ratio6 / LJcutoff;
  cudaMemcpyToSymbol(d_LJfshift, &LJfshift, sizeof(LJfshift));
  // repulsive Lennard-Jones
  LJecutPlus = 4 * ec * (ratio6 * ratio6 + ratio6);
  cudaMemcpyToSymbol(d_LJecutPlus, &LJecutPlus, sizeof(LJecutPlus));
	LJfshiftPlus = 24 * ec * (2 * ratio6 + 1) * ratio6 / LJcutoff;
  cudaMemcpyToSymbol(d_LJfshiftPlus, &LJfshiftPlus, sizeof(LJfshiftPlus));
  cout << "SP2D::setLJMinusPlusParams::LJcutoff: " << LJcutoff << " energy shift: " << LJecut << " LJfshift: " << LJfshift << endl;
  num1 = num1_;
  cudaMemcpyToSymbol(d_num1, &num1, sizeof(num1));
  long num1FromDevice = 0;
  cudaMemcpyFromSymbol(&num1FromDevice, d_num1, sizeof(d_num1));
  cout << "SP2D::setLJMinusPlusParams::num1: " << num1 << " from device: " << num1FromDevice << endl;
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

void SP2D::calcParticleInteraction() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  //cout << "dimGrid, dimBlock: " << dimGrid << ", " << dimBlock << endl;
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    kernelCalcParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
    case simControlStruct::neighborEnum::allToAll:
    kernelCalcAllToAllParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
    break;
    default:
    break;
  }
}

void SP2D::addSelfPropulsion() {
  int s_nDim(nDim);
  double s_driving(driving);
  double amplitude = sqrt(2.0 * dt / taup);
  auto r = thrust::counting_iterator<long>(0);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  double *pAngle = thrust::raw_pointer_cast(&(d_particleAngle[0]));
  double* pForce = thrust::raw_pointer_cast(&(d_particleForce[0]));
	if(nDim == 2) {
    thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_activeAngle.begin(), gaussNum(0.f,1.f));
    const double *activeAngle = thrust::raw_pointer_cast(&d_activeAngle[0]);

    auto langevinUpdateActiveNoise2D = [=] __device__ (long pId) {
      pAngle[pId] += amplitude * activeAngle[pId];
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pForce[pId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId]));
      }
    };

    thrust::for_each(r, r + numParticles, langevinUpdateActiveNoise2D);

  } else if(nDim == 3) {
    auto s = thrust::counting_iterator<long>(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + numParticles * nDim, d_activeAngle.begin(), gaussNum(0.f,1.f));
    double *activeAngle = thrust::raw_pointer_cast(&d_activeAngle[0]);

    auto normalizeVector = [=] __device__ (long particleId) {
      auto norm = 0.0;
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < nDim; dim++) {
        norm += activeAngle[particleId * s_nDim + dim] * activeAngle[particleId * s_nDim + dim];
      }
      norm = sqrt(norm);
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        activeAngle[particleId * s_nDim + dim] /= norm;
      }
    };

    thrust::for_each(s, s + numParticles, normalizeVector);

    auto langevinUpdateActiveNoise3D = [=] __device__ (long particleId) {
      pAngle[particleId * s_nDim] += amplitude * (pAngle[particleId * s_nDim + 1] * activeAngle[particleId * s_nDim + 2] - pAngle[particleId * s_nDim + 2] * activeAngle[particleId * s_nDim + 1]);
      pAngle[particleId * s_nDim + 1] += amplitude * (pAngle[particleId * s_nDim + 2] * activeAngle[particleId * s_nDim] - pAngle[particleId * s_nDim] * activeAngle[particleId * s_nDim + 2]);
      pAngle[particleId * s_nDim + 2] += amplitude * (pAngle[particleId * s_nDim] * activeAngle[particleId * s_nDim + 1] - pAngle[particleId * s_nDim + 1] * activeAngle[particleId * s_nDim]);
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pForce[particleId * s_nDim + dim] += driving * pAngle[particleId * s_nDim + dim];
      }
    };

    thrust::for_each(r, r + numParticles, langevinUpdateActiveNoise3D);
  }
}

void SP2D::addParticleWallInteraction() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
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
    default:
    break;
	}
}

void SP2D::addParticleGravity() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  kernelAddParticleGravity<<<dimGrid, dimBlock>>>(pPos, pForce, pEnergy);
}

void SP2D::calcParticleForceEnergy() {
  calcParticleInteraction();
  if(simControl.particleType == simControlStruct::particleEnum::active) {
    addSelfPropulsion();
  }
  if(simControl.geometryType != simControlStruct::geometryEnum::normal) {
    addParticleWallInteraction();
  }
  switch (simControl.gravityType) {
    case simControlStruct::gravityEnum::on:
    addParticleGravity();
    break;
    default:
    break;
  }
}

void SP2D::setTwoParticleTestPacking(double sigma0, double sigma1, double lx, double ly, double y0, double y1, double vel1) {
  thrust::host_vector<double> boxSize(nDim);
  // set particle radii
  d_particleRad[0] = 0.5 * sigma0;
  d_particleRad[1] = 0.5 * sigma1;
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  // assign positions
  for (int pId = 0; pId < numParticles; pId++) {
    d_particlePos[pId * nDim] = lx * 0.5;
  }
  d_particlePos[0 * nDim + 1] = ly * y0;
  d_particlePos[1 * nDim + 1] = ly * y1;
  // assign velocity
  d_particleVel[1 * nDim + 1] = vel1;
  setLengthScaleToOne();
}

void SP2D::setThreeParticleTestPacking(double sigma02, double sigma1, double lx, double ly, double y02, double y1, double vel1) {
  thrust::host_vector<double> boxSize(nDim);
  // set particle radii
  d_particleRad[0] = 0.5 * sigma02;
  d_particleRad[1] = 0.5 * sigma1;
  d_particleRad[2] = 0.5 * sigma02;
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  // assign positions
  d_particlePos[0 * nDim] = lx * 0.35;
  d_particlePos[0 * nDim + 1] = ly * y02;
  d_particlePos[2 * nDim] = lx * 0.65;
  d_particlePos[2 * nDim + 1] = ly * y02;
  d_particlePos[1 * nDim] = lx * 0.5;
  d_particlePos[1 * nDim + 1] = ly * y1;
  // assign velocity
  d_particleVel[1 * nDim + 1] = vel1;
  setLengthScaleToOne();
}

void SP2D::firstUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	const double* pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

  auto firstUpdate = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += 0.5 * s_dt * pForce[pId * s_nDim + dim];
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + numParticles, firstUpdate);
}

void SP2D::secondUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	const double* pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

  auto firstUpdate = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += 0.5 * s_dt * pForce[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + numParticles, firstUpdate);
}

void SP2D::testInteraction(double timeStep) {
  firstUpdate(timeStep);
  checkParticleNeighbors();
  calcParticleForceEnergy();
  secondUpdate(timeStep);
}

void SP2D::printTwoParticles() {
  cudaDeviceSynchronize();
  if(cudaGetLastError) cout << "SP2D::printTwoParticles:: cudaGetLastError(): " << cudaGetLastError() << endl;
  thrust::host_vector<double> particleForce(d_particleForce.size(), 0.0);
  thrust::host_vector<double> particleVel(d_particleForce.size(), 0.0);
  thrust::host_vector<double> particlePos(d_particleForce.size(), 0.0);
  particleForce = d_particleForce;
  particleVel = d_particleVel;
  particlePos = d_particlePos;
  cout << "particle 0: fx: " << particleForce[0] << " fy: " << particleForce[1] << endl;
  cout << "particle 0: vx: " << particleVel[0] << " vy: " << particleVel[1] << endl;
  cout << "particle 0: x: " << particlePos[0] << " y: " << particlePos[1] << endl;
  cout << "particle 1: fx: " << particleForce[2] << " fy: " << particleForce[3] << endl;
  cout << "particle 1: vx: " << particleVel[2] << " vy: " << particleVel[3] << endl;
  cout << "particle 1: x: " << particlePos[2] << " y: " << particlePos[3] << endl;
}

void SP2D::printThreeParticles() {
  cudaDeviceSynchronize();
  if(cudaGetLastError) cout << "SP2D::printThreeParticles:: cudaGetLastError(): " << cudaGetLastError() << endl;
  thrust::host_vector<double> particleForce(d_particleForce.size(), 0.0);
  thrust::host_vector<double> particleVel(d_particleForce.size(), 0.0);
  thrust::host_vector<double> particlePos(d_particleForce.size(), 0.0);
  particleForce = d_particleForce;
  particleVel = d_particleVel;
  particlePos = d_particlePos;
  cout << "particle 0: fx: " << particleForce[0] << " fy: " << particleForce[1] << endl;
  cout << "particle 0: vx: " << particleVel[0] << " vy: " << particleVel[1] << endl;
  cout << "particle 0: x: " << particlePos[0] << " y: " << particlePos[1] << endl;
  cout << "particle 1: fx: " << particleForce[2] << " fy: " << particleForce[3] << endl;
  cout << "particle 1: vx: " << particleVel[2] << " vy: " << particleVel[3] << endl;
  cout << "particle 1: x: " << particlePos[2] << " y: " << particlePos[3] << endl;
  cout << "particle 2: fx: " << particleForce[4] << " fy: " << particleForce[5] << endl;
  cout << "particle 2: vx: " << particleVel[4] << " vy: " << particleVel[5] << endl;
  cout << "particle 2: x: " << particlePos[4] << " y: " << particlePos[5] << endl;
}

 void SP2D::makeExternalParticleForce(double externalForce) {
   // extract +-1 random forces
   d_particleDelta.resize(numParticles);
   thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
   thrust::counting_iterator<long> index_sequence_begin(0);
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
  kernelCalcStressTensor<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, pStress);
}

double SP2D::getParticlePressure() {
  calcParticleStressTensor();
  double volume = 1.0;
  double stress = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
    stress += d_stress[dim * nDim + dim];
  }
  return stress / (nDim * volume);
}

void SP2D::calcParticleActiveStressTensor() {
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *pStress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcActiveStress<<<dimGrid, dimBlock>>>(pAngle, pVel, pStress);
}

double SP2D::getParticleActivePressure() {
  calcParticleActiveStressTensor();
  double volume = 1.0;
  double stress = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
    stress += d_stress[dim * nDim + dim];
  }
  return stress / (nDim * volume);
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

std::tuple<double, double, double> SP2D::getParticleStressComponents() {
   calcParticleStressTensor();
   double stress_xx = d_stress[0] / (d_boxSize[0] * d_boxSize[1]);
   double stress_yy = d_stress[3] / (d_boxSize[0] * d_boxSize[1]);
   double stress_xy = (d_stress[1] + d_stress[2]) / (nDim * d_boxSize[0] * d_boxSize[1]);
   return std::make_tuple(stress_xx, stress_yy, stress_xy);
}

double SP2D::getParticleWallPressure() {
  thrust::device_vector<double> d_wallStress(d_particleRad.size());
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wallStress = thrust::raw_pointer_cast(&d_wallStress[0]);
  kernelCalcBoxStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
  double length = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    length += 2 * d_boxSize[dim];
  }
	return thrust::reduce(d_wallStress.begin(), d_wallStress.end(), double(0), thrust::plus<double>()) / length;
}

double SP2D::getParticleBoxPressure() {
  thrust::device_vector<double> d_wallStress(d_particleRad.size());
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wallStress = thrust::raw_pointer_cast(&d_wallStress[0]);
  switch (simControl.geometryType) {
    case simControlStruct::geometryEnum::normal:
    break;
		case simControlStruct::geometryEnum::fixedBox:
    kernelCalcBoxStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
    kernelCalcSides2DStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
    break;
    default:
    break;
	}
  double length = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    length += 2 * d_boxSize[dim];
  }
	return thrust::reduce(d_wallStress.begin(), d_wallStress.end(), double(0), thrust::plus<double>()) / length;
}

std::tuple<double, double> SP2D::getColumnWork(double width) {
  double workIn = 0.0;
  double workOut = 0.0;
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  kernelCalcColumnWork<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, width, workIn, workOut);
  return std::make_tuple(workIn, workOut);
}

std::tuple<double, double> SP2D::getColumnActiveWork(double width) {
  double activeWorkIn = 0.0;
  double activeWorkOut = 0.0;
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  kernelCalcColumnActiveWork<<<dimGrid, dimBlock>>>(pPos, pAngle, pVel, width, activeWorkIn, activeWorkOut);
  activeWorkIn *= (driving * taup * 0.5);
  activeWorkOut *= (driving * taup * 0.5);
  return std::make_tuple(activeWorkIn, activeWorkOut);
}

double SP2D::getParticleWallForce(double range, double width) {
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
  if(width == 0.0) {
    kernelCalcWallForce<<<dimGrid, dimBlock>>>(pRad, pPosPBC, range, wallForce, wallCount);
    if(simControl.particleType == simControlStruct::particleEnum::active) {
      const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
      kernelAddWallActiveForce<<<dimGrid, dimBlock>>>(pAngle, wallForce, wallCount);
    }
  } else {
    kernelCalcCenterWallForce<<<dimGrid, dimBlock>>>(pRad, pPosPBC, range, width, wallForce, wallCount);
  }
  return thrust::reduce(d_wallForce.begin(), d_wallForce.end(), double(0), thrust::plus<double>());
}

long SP2D::getTotalParticleWallCount() {
  return thrust::reduce(d_wallCount.begin(), d_wallCount.end(), long(0), thrust::plus<long>());
}

double SP2D::getParticlePotentialEnergy() {
  return thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleKineticEnergy() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
}

double SP2D::getDampingWork() {
  thrust::device_vector<double> d_dampingWork(d_particleEnergy.size());
  thrust::fill(d_dampingWork.begin(), d_dampingWork.end(), double(0));

  double s_dt(dt);
  long s_nDim(nDim);
  double s_gamma(this->sim_->gamma);
  auto r = thrust::counting_iterator<long>(0);
	const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double* dWork = thrust::raw_pointer_cast(&d_dampingWork[0]);

  auto computeDampingWork = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      dWork[pId] -= s_dt * s_gamma * pVel[pId * s_nDim + dim] * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + numParticles, computeDampingWork);

  return thrust::reduce(d_dampingWork.begin(), d_dampingWork.end(), double(0), thrust::plus<double>());
}

double SP2D::getSelfPropulsionWork() {
  thrust::device_vector<double> d_activeWork(d_particleEnergy.size());
  thrust::fill(d_activeWork.begin(), d_activeWork.end(), double(0));

  double s_dt(dt);
  long s_nDim(nDim);
  double s_driving(driving);
  auto r = thrust::counting_iterator<long>(0);
	const double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
	const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double* aWork = thrust::raw_pointer_cast(&d_activeWork[0]);

  auto computeActiveWork = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      aWork[pId] += s_dt * s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId])) * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + numParticles, computeActiveWork);

  return thrust::reduce(d_activeWork.begin(), d_activeWork.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleTemperature() {
  return 2 * getParticleKineticEnergy() / (nDim * numParticles);
}

double SP2D::getParticleEnergy() {
  return getParticlePotentialEnergy() + getParticleKineticEnergy();
}

double SP2D::getParticleWork() {
  double work = getDampingWork();
  if(simControl.particleType == simControlStruct::particleEnum::active) {
    work += getSelfPropulsionWork();
  }
  return work;
}

std::tuple<double, double, double> SP2D::getParticleKineticEnergy12() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  thrust::device_vector<double> velSq1(num1 * nDim);
  thrust::device_vector<double> velSq2((numParticles-num1) * nDim);
  thrust::copy(velSquared.begin(), velSquared.begin() + num1 * nDim, velSq1.begin());
  thrust::copy(velSquared.begin() + num1 * nDim, velSquared.end(), velSq2.begin());
  double ekin1 = 0.5 * thrust::reduce(velSq1.begin(), velSq1.end(), double(0), thrust::plus<double>());
  double ekin2 = 0.5 * thrust::reduce(velSq2.begin(), velSq2.end(), double(0), thrust::plus<double>());
  double ekin = 0.5 * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
  return std::make_tuple(ekin1, ekin2, ekin);
}

std::tuple<double, double, double> SP2D::getParticleT1T2() {
  std::tuple<double, double, double> ekins = getParticleKineticEnergy12();
  double T1 = 2 * get<0>(ekins) / (nDim * num1);
  double T2 = 2 * get<1>(ekins) / (nDim * (numParticles - num1));
  double T = 2 * get<2>(ekins) / (nDim * numParticles);
  //double T = 2 * getParticleKineticEnergy() / (nDim * numParticles);
  return std::make_tuple(T1, T2, T);
}

void SP2D::adjustKineticEnergy(double prevEtot) {
  double scale, ekin = getParticleKineticEnergy();
  double deltaEtot = getParticlePotentialEnergy() + ekin;
  deltaEtot -= prevEtot;
  if(ekin > deltaEtot) {
    scale = sqrt((ekin - deltaEtot) / ekin);
    //cout << "deltaEtot: " << deltaEtot << " ekin - deltaEtot: " << ekin - deltaEtot << " scale: " << scale << endl;
    long s_nDim(nDim);
    auto r = thrust::counting_iterator<long>(0);
    double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

    auto adjustParticleVel = [=] __device__ (long pId) {
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pVel[pId * s_nDim + dim] *= scale;
      }
    };

    cout << "SP2D::adjustKineticEnergy:: scale: " << scale << endl;
    thrust::for_each(r, r + numParticles, adjustParticleVel);
  } else {
    cout << "SP2D::adjustKineticEnergy:: kinetic energy is less then change in total energy - no adjustment is made" << endl;
  }
}

void SP2D::adjustLocalKineticEnergy(thrust::host_vector<double> &prevEnergy_) {
  thrust::device_vector<double> d_prevEnergy = prevEnergy_;
  // compute new potential energy per particle
  getParticlePotentialEnergy();
  // locally rescale velocities
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  const double *prevEnergy = thrust::raw_pointer_cast(&d_prevEnergy[0]);

  auto adjustLocalParticleVel = [=] __device__(long pId) {
    double deltaU = pEnergy[pId] - prevEnergy[pId];
    double ekin = 0.0;
    #pragma unroll(MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      ekin += pVel[pId * s_nDim + dim] * pVel[pId * s_nDim + dim];
    }
    ekin *= 0.5;
    if(ekin > deltaU) {
      double scale = sqrt((ekin - deltaU) / ekin);
      #pragma unroll(MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pVel[pId * s_nDim + dim] *= scale;
      }
    }
    //double scale = 1.0;
    //if(ekin > deltaU) {
    //  scale = sqrt((ekin - deltaU) / ekin);
    //} else {
    //  scale = sqrt((deltaU - ekin) / ekin);
    //}
    //#pragma unroll(MAXDIM)
    //for (long dim = 0; dim < s_nDim; dim++) {
    //  pVel[pId * s_nDim + dim] *= scale;
    //}
  };

  thrust::for_each(r, r + numParticles, adjustLocalParticleVel);
}

void SP2D::adjustTemperature(double targetTemp) {
  double scale = sqrt(targetTemp / getParticleTemperature());
  //cout << "deltaEtot: " << deltaEtot << " ekin - deltaEtot: " << ekin - deltaEtot << " scale: " << scale << endl;
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

  auto adjustParticleTemp = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] *= scale;
    }
  };

  //cout << "SP2D::adjustTemperature:: scale: " << scale << endl;
  thrust::for_each(r, r + numParticles, adjustParticleTemp);
}

double SP2D::getMassiveTemperature(long firstIndex, double mass) {
  // temperature computed from the massive particles which are set to be the first
  thrust::device_vector<double> velSquared(firstIndex * nDim);
  thrust::transform(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, velSquared.begin(), square());
  return mass * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>()) / (firstIndex * nDim);
}

//************************* contacts and neighbors ***************************//
thrust::host_vector<long> SP2D::getParticleNeighbors() {
  thrust::host_vector<long> partNeighborListFromDevice;
  partNeighborListFromDevice = d_partNeighborList;
  return partNeighborListFromDevice;
}

void SP2D::calcParticleNeighbors(double cutDistance) {
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    calcParticleNeighborList(cutDistance);
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
    default:
    break;
  }
}

void SP2D::calcParticleNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  if(cudaGetLastError()) cout << "SP2D::calcParticleNeighborList():: cudaGetLastError(): " << cudaGetLastError() << endl;
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
  cudaDeviceSynchronize();
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_partNeighborListSize, &partNeighborListSize, sizeof(partNeighborListSize));
	cudaMemcpyToSymbol(d_partMaxNeighbors, &partMaxNeighbors, sizeof(partMaxNeighbors));

	long* partMaxNeighborList = thrust::raw_pointer_cast(&d_partMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_partMaxNeighborListPtr, &partMaxNeighborList, sizeof(partMaxNeighborList));

	long* partNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
	cudaMemcpyToSymbol(d_partNeighborListPtr, &partNeighborList, sizeof(partNeighborList));
  if(cudaGetLastError()) cout << "SP2D::syncParticleNeighborsToDevice():: cudaGetLastError(): " << cudaGetLastError() << endl;
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
  setInitialPositions();
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
  setInitialPositions();
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
  setInitialPositions();
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
  setInitialPositions();
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
  setInitialPositions();
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
  setInitialPositions();
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
  setInitialPositions();
  shift = true;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleNVE:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleNVELoop() {
  this->sim_->integrate();
}

//******************* NVE integrator with velocity rescale *******************//
void SP2D::initSoftParticleNVERescale(double Temp) {
  this->sim_ = new SoftParticleNVERescale(this, SimConfig(Temp, 0, 0));
  resetLastPositions();
  setInitialPositions();
  shift = true;
  cout << "SP2D::initSoftParticleNVERescale:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleNVERescaleLoop() {
  this->sim_->integrate();
}

//*************** NVE integrator with double velocity rescale ****************//
void SP2D::initSoftParticleNVEDoubleRescale(double Temp1, double Temp2) {
  this->sim_ = new SoftParticleNVEDoubleRescale(this, SimConfig(Temp1, 0, Temp2));
  resetLastPositions();
  setInitialPositions();
  shift = true;
  std::tuple<double, double, double> Temps = getParticleT1T2();
  cout << "SP2D::initSoftParticleNVEDoubleRescale:: T1: " << setprecision(12) << get<0>(Temps) << " T2: " << get<1>(Temps) << " T: " << get<2>(Temps) << endl;
}

void SP2D::softParticleNVEDoubleRescaleLoop() {
  this->sim_->integrate();
}

void SP2D::getNoseHooverParams(double &mass, double &damping) {
  mass = this->sim_->mass;
  damping = this->sim_->gamma;
  //cout << "SP2D::getNoseHooverParams:: damping: " << this->sim_->gamma << endl;
}

//************************* Nose-Hoover integrator ***************************//
void SP2D::initSoftParticleNoseHoover(double Temp, double mass, double gamma, bool readState) {
  this->sim_ = new SoftParticleNoseHoover(this, SimConfig(Temp, 0, 0));
  this->sim_->mass = mass;
  this->sim_->gamma = gamma;
  resetLastPositions();
  setInitialPositions();
  shift = true;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "SP2D::initSoftParticleNoseHoover:: current temperature: " << setprecision(12) << getParticleTemperature();
  cout << " mass: " << this->sim_->mass << ", damping: " << this->sim_->gamma << endl;
}

void SP2D::softParticleNoseHooverLoop() {
  this->sim_->integrate();
}

void SP2D::getDoubleNoseHooverParams(double &mass, double &damping1, double &damping2) {
  mass = this->sim_->mass;
  damping1 = this->sim_->lcoeff1;
  damping2 = this->sim_->lcoeff2;
  //cout << "SP2D::getNoseHooverParams:: damping: " << this->sim_->gamma << endl;
}

//********************** double T Nose-Hoover integrator *********************//
void SP2D::initSoftParticleDoubleNoseHoover(double Temp1, double Temp2, double mass, double gamma1, double gamma2, bool readState) {
  this->sim_ = new SoftParticleDoubleNoseHoover(this, SimConfig(Temp1, 0, Temp2));
  this->sim_->mass = mass;
  this->sim_->lcoeff1 = gamma1;
  this->sim_->lcoeff2 = gamma2;
  resetLastPositions();
  setInitialPositions();
  shift = true;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  std::tuple<double, double, double> Temps = getParticleT1T2();
  cout << "SP2D::initSoftParticleDoubleNoseHoover:: T1: " << setprecision(12) << get<0>(Temps) << " T2: " << get<1>(Temps) << " T: " << get<2>(Temps) << endl;
  cout << " mass: " << this->sim_->mass << ", damping1: " << this->sim_->lcoeff1 << " damping2: " << this->sim_->lcoeff2 << endl;
}

void SP2D::softParticleDoubleNoseHooverLoop() {
  this->sim_->integrate();
}
