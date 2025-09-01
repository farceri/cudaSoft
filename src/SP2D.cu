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
  nDim = dim;
  numParticles = nParticles;
  num1 = numParticles;
  if(numParticles < 256) {
    dimBlock = 32;
  } else {
    dimBlock = 256;
  }
  setDimBlock(dimBlock);
  setNDim(nDim);
  setNumParticles(numParticles);
	simControl.particleType = simControlStruct::particleEnum::passive;
	simControl.noiseType = simControlStruct::noiseEnum::langevin2;
	simControl.boundaryType = simControlStruct::boundaryEnum::pbc;
	simControl.geometryType = simControlStruct::geometryEnum::squareWall;
	simControl.neighborType = simControlStruct::neighborEnum::neighbor;
	simControl.potentialType = simControlStruct::potentialEnum::harmonic;
	simControl.wallType = simControlStruct::wallEnum::harmonic;
	simControl.gravityType = simControlStruct::gravityEnum::off;
	simControl.alignType = simControlStruct::alignEnum::additive;
	syncSimControlToDevice();
  // default parameters
  dt = 1e-04;
  rho0 = 1;
	ec = 1;
	l1 = 0;
	l2 = 0;
  LEshift = 0;
  gravity = 0;
  ew = 1;
  flowSpeed = 0;
  flowDecay = 1;
  flowViscosity = 1;
  cutDistance = 1;
  updateCount = 0;
  shift = false;
  ea = 1e05;
  el = 1;
  eb = 1;
  angleAmplitude = 0.0;
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
  d_squaredVel.resize(numParticles_ * nDim);
  thrust::fill(d_particleRad.begin(), d_particleRad.end(), double(0));
  thrust::fill(d_particlePos.begin(), d_particlePos.end(), double(0));
  thrust::fill(d_particleVel.begin(), d_particleVel.end(), double(0));
  thrust::fill(d_particleForce.begin(), d_particleForce.end(), double(0));
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  thrust::fill(d_squaredVel.begin(), d_squaredVel.end(), double(0));
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
  d_flag.resize(numParticles_);
  thrust::fill(d_flag.begin(), d_flag.end(), int(0));
}

void SP2D::initVicsekNeighbors(long numParticles_) {
  vicsekNeighborListSize = 0;
  vicsekMaxNeighbors = 0;
  d_vicsekNeighborList.resize(numParticles_);
  d_vicsekMaxNeighborList.resize(numParticles_);
  thrust::fill(d_vicsekNeighborList.begin(), d_vicsekNeighborList.end(), -1L);
  thrust::fill(d_vicsekMaxNeighborList.begin(), d_vicsekMaxNeighborList.end(), vicsekMaxNeighbors);
  d_vicsekFlag.resize(numParticles_);
  thrust::fill(d_vicsekFlag.begin(), d_vicsekFlag.end(), int(0));
}

void SP2D::initWallVariables(long numWall_) {
  d_wallPos.resize(numWall_ * nDim);
  d_wallForce.resize(numWall_ * nDim);
  d_wallEnergy.resize(numWall_);
  thrust::fill(d_wallPos.begin(), d_wallPos.end(), double(0));
  thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
  thrust::fill(d_wallEnergy.begin(), d_wallEnergy.end(), double(0));
  if(simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    d_monomerAlpha.resize(numWall_);
    thrust::fill(d_monomerAlpha.begin(), d_monomerAlpha.end(), double(0));
    cout << "SP2D::initWallVariables: boundaryType: rigid" << endl;
    wallAngle = 0.;
    wallOmega = 0.;
    wallAlpha = 0.;
  }
}

void SP2D::initWallShapeVariables(long numWall_) {
  d_wallLength.resize(numWall_);
  d_wallAngle.resize(numWall_);
  d_areaSector.resize(numWall_);
  d_wallVel.resize(numWall_ * nDim);
  d_sqWallVel.resize(numWall_ * nDim);
  thrust::fill(d_wallLength.begin(), d_wallLength.end(), double(0));
  thrust::fill(d_wallAngle.begin(), d_wallAngle.end(), double(0));
  thrust::fill(d_areaSector.begin(), d_areaSector.end(), double(0));
  thrust::fill(d_wallVel.begin(), d_wallVel.end(), double(0));
  thrust::fill(d_sqWallVel.begin(), d_sqWallVel.end(), double(0));
  if(simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    d_restLength.resize(numWall_);
    thrust::fill(d_restLength.begin(), d_restLength.end(), double(0));
  }
}

void SP2D::initWallNeighbors(long numParticles_) {
  wallNeighborListSize = 0;
  wallMaxNeighbors = 0;
  d_wallNeighborList.resize(numParticles_);
  d_wallMaxNeighborList.resize(numParticles_);
  thrust::fill(d_wallNeighborList.begin(), d_wallNeighborList.end(), -1L);
  thrust::fill(d_wallMaxNeighborList.begin(), d_wallMaxNeighborList.end(), wallMaxNeighbors);
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
      d_particleAngle.resize(numParticles);
      d_randAngle.resize(numParticles);
    } else if(nDim == 3) {
      d_particleAngle.resize(numParticles * nDim);
      d_randAngle.resize(numParticles * nDim);
    }
    thrust::fill(d_particleAngle.begin(), d_particleAngle.end(), double(0));
    thrust::fill(d_randAngle.begin(), d_randAngle.end(), double(0));
    d_velCorr.resize(numParticles);
    thrust::fill(d_velCorr.begin(), d_velCorr.end(), double(0));
    d_angMom.resize(numParticles);
    thrust::fill(d_angMom.begin(), d_angMom.end(), double(0));
    cout << "SP2D::setParticleType: particleType: active" << endl;
  } else if(simControl.particleType == simControlStruct::particleEnum::vicsek) {
    d_randAngle.resize(numParticles);
    d_particleAngle.resize(numParticles);
    d_particleAlpha.resize(numParticles);
    thrust::fill(d_randAngle.begin(), d_randAngle.end(), double(0));
    thrust::fill(d_particleAngle.begin(), d_particleAngle.end(), double(0));
    thrust::fill(d_particleAlpha.begin(), d_particleAlpha.end(), double(0));
    initVicsekNeighbors(numParticles);
    d_vicsekLastPos.resize(numParticles * nDim);
    thrust::fill(d_vicsekLastPos.begin(), d_vicsekLastPos.end(), double(0));
    d_velCorr.resize(numParticles);
    thrust::fill(d_velCorr.begin(), d_velCorr.end(), double(0));
    d_angMom.resize(numParticles);
    thrust::fill(d_angMom.begin(), d_angMom.end(), double(0));
    d_unitPos.resize(numParticles * nDim);
    thrust::fill(d_unitPos.begin(), d_unitPos.end(), double(0));
    d_unitVel.resize(numParticles * nDim);
    thrust::fill(d_unitVel.begin(), d_unitVel.end(), double(0));
    d_unitVelPos.resize(numParticles * nDim);
    thrust::fill(d_unitVelPos.begin(), d_unitVelPos.end(), double(0));
    cout << "SP2D::setParticleType: particleType: vicsek" << endl;
  } else {
    cout << "SP2D::setParticleType: please specify valid particleType: passive, active or vicsek" << endl;
  }
	syncSimControlToDevice();
}

void SP2D::setNoiseType(simControlStruct::noiseEnum noiseType_) {
	simControl.noiseType = noiseType_;
  if(simControl.noiseType == simControlStruct::noiseEnum::langevin1) {
    cout << "SP2D::setNoiseType: noiseType: langevin1" << endl;
  } else if(simControl.noiseType == simControlStruct::noiseEnum::langevin2) {
    cout << "SP2D::setNoiseType: noiseType: langevin2" << endl;
  } else if(simControl.noiseType == simControlStruct::noiseEnum::brownian) {
    cout << "SP2D::setNoiseType: noiseType: brownian" << endl;
  } else if(simControl.noiseType == simControlStruct::noiseEnum::drivenBrownian) {
    cout << "SP2D::setNoiseType: noiseType: drivenBrownian" << endl;
  } else {
    cout << "SP2D::setNoiseType: please specify valid particleType: langevin1, langevin2, drivenBrownian and drivenLangevin" << endl;
  }
	syncSimControlToDevice();
}

void SP2D::setBoundaryType(simControlStruct::boundaryEnum boundaryType_) {
	simControl.boundaryType = boundaryType_;
  if(simControl.boundaryType == simControlStruct::boundaryEnum::pbc) {
    cout << "SP2D::setBoundaryType: boundaryType: pbc" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::leesEdwards) {
    cout << "SP2D::setBoundaryType: boundaryType: leesEdwards" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::fixed) {
    cout << "SP2D::setBoundaryType: boundaryType: fixed" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::reflect) {
    cout << "SP2D::setBoundaryType: boundaryType: reflect" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::reflectNoise) {
    d_randomAngle.resize(numParticles);
    thrust::fill(d_randomAngle.begin(), d_randomAngle.end(), double(0));
    cout << "SP2D::setBoundaryType: boundaryType: reflectnoise" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::rough) {
    setNeighborType(simControlStruct::neighborEnum::neighbor);
    setGeometryType(simControlStruct::geometryEnum::roundWall);
    cout << "SP2D::setBoundaryType: boundaryType: rough" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    setNeighborType(simControlStruct::neighborEnum::neighbor);
    setGeometryType(simControlStruct::geometryEnum::roundWall);
    cout << "SP2D::setBoundaryType: boundaryType: rigid" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::mobile) {
    setNeighborType(simControlStruct::neighborEnum::neighbor);
    setGeometryType(simControlStruct::geometryEnum::roundWall);
    cout << "SP2D::setBoundaryType: boundaryType: mobile" << endl;
  } else if(simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    setNeighborType(simControlStruct::neighborEnum::neighbor);
    setGeometryType(simControlStruct::geometryEnum::roundWall);
    lgamma = 1.;
    cout << "SP2D::setBoundaryType: boundaryType: plastic" << endl;
  } else {
    cout << "SP2D::setBoundaryType: please specify valid boundaryType: pbc, leesEdwards, fixed, reflect, reflectNoise, rough, rigid, mobile and plastic" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::boundaryEnum SP2D::getBoundaryType() {
	syncSimControlFromDevice();
	return simControl.boundaryType;
}

void SP2D::setGeometryType(simControlStruct::geometryEnum geometryType_) {
	simControl.geometryType = geometryType_;
  if(simControl.geometryType == simControlStruct::geometryEnum::squareWall) {
    d_wallForce.resize(numParticles * nDim);
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    cout << "SP2D::setGeometryType: geometryType: squareWall" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides2D) {
    setBoundaryType(simControlStruct::boundaryEnum::fixed);
    d_wallForce.resize(numParticles * nDim);
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    cout << "SP2D:;setGeometryType: geometryType: fixedSides2D" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::fixedSides3D) {
    setBoundaryType(simControlStruct::boundaryEnum::fixed);
    d_wallForce.resize(numParticles * nDim);
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    cout << "SP2D::setGeometryType: geometryType: fixedSides3D" << endl;
  } else if(simControl.geometryType == simControlStruct::geometryEnum::roundWall) {
    d_wallForce.resize(numParticles * nDim);
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    cout << "SP2D::setGeometryType: geometryType: roundWall" << endl;
  } else {
    cout << "SP2D::setGeometryType: please specify valid geometryType: squareWall, fixedSides2D, fixedSides3D and roundWall" << endl;
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
  if(simControl.potentialType == simControlStruct::potentialEnum::none) {
    setBoundaryType(simControlStruct::boundaryEnum::reflect);
    cout << "SP2D::setPotentialType: potentialType: none" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::harmonic) {
    cout << "SP2D::setPotentialType: potentialType: harmonic" << " set wallType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::lennardJones) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: lennardJones" << " set wallType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::Mie) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: Mie" << " set wallType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::WCA) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: WCA" << " set wallType: WCA" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::adhesive) {
    cout << "SP2D::setPotentialType: potentialType: adhesive" << " set wallType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::doubleLJ) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: doubleLJ" << " set wallType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::LJMinusPlus) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: LJMinusPlus" << " set wallType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::LJWCA) {
    setWallType(simControlStruct::wallEnum::WCA);
    cout << "SP2D::setPotentialType: potentialType: LJWCA" << " set wallType: lennardJones" << endl;
  } else {
    cout << "SP2D::setPotentialType: please specify valid potentialType: none, harmonic, lennardJones, WCA, adhesive, doubleLJ, LJMinusPlus and LJWCA" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::potentialEnum SP2D::getPotentialType() {
	syncSimControlFromDevice();
	return simControl.potentialType;
}

void SP2D::setWallType(simControlStruct::wallEnum wallType_) {
	simControl.wallType = wallType_;
  if(simControl.wallType == simControlStruct::wallEnum::harmonic) {
    cout << "SP2D::setWallType: wallType: harmonic" << endl;
  } else if(simControl.wallType == simControlStruct::wallEnum::lennardJones) {
    cout << "SP2D::setWallType: wallType: lennardJones" << endl;
  } else if(simControl.wallType == simControlStruct::wallEnum::WCA) {
    cout << "SP2D::setWallType: wallType: WCA" << endl;
  } else {
    cout << "SP2D::setWallType: please specify valid wallType: harmonic, lennardJones and  WCA" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::wallEnum SP2D::getWallType() {
	syncSimControlFromDevice();
	return simControl.wallType;
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

void SP2D::setAlignType(simControlStruct::alignEnum alignType_) {
	simControl.alignType = alignType_;
  if(simControl.alignType == simControlStruct::alignEnum::additive) {
    cout << "SP2D::setAlignType: alignType: additive" << endl;
  } else if(simControl.alignType == simControlStruct::alignEnum::nonAdditive) {
    cout << "SP2D::setAlignType: alignType: non additive" << endl;
  } else if(simControl.alignType == simControlStruct::alignEnum::velAlign) {
    cout << "SP2D::setAlignType: alignType: velocity" << endl;
  } else {
    cout << "SP2D::setAlignType: please specify valid alignType: additive or non additive" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::alignEnum SP2D::getAlignType() {
	syncSimControlFromDevice();
	return simControl.alignType;
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
	if(simControl.boundaryType == simControlStruct::boundaryEnum::leesEdwards) {
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
  thrust::host_vector<double> newWallSize(nDim);
  newWallSize = getBoxSize();
  newWallSize[1] = (1 + strainy_) * newWallSize[1];
  setBoxSize(newWallSize);
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

void SP2D::applyUniaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newWallSize_);
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

void SP2D::applyCenteredUniaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newWallSize_);
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

void SP2D::applyBiaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newWallSize_);
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

void SP2D::applyBiaxialExpExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newWallSize_);
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

void SP2D::applyCenteredBiaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_) {
  // first set the new boxSize
  setBoxSize(newWallSize_);
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
  cudaError err = cudaMemcpyToSymbol(d_dimBlock, &dimBlock, sizeof(dimBlock));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
	dimGrid = (numParticles + dimBlock - 1) / dimBlock;
  err = cudaMemcpyToSymbol(d_dimGrid, &dimGrid, sizeof(dimGrid));
  cout << "SP2D::setDimBlock: dimBlock " << dimBlock << " dimGrid " << dimGrid << endl;
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
}

void SP2D::setWallBlock(long dimBlock_) {
	dimBlock = dimBlock_;
  cudaError err = cudaMemcpyToSymbol(d_dimBlock, &dimBlock, sizeof(dimBlock));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
  dimGrid = (numWall + dimBlock - 1) / dimBlock;
  err = cudaMemcpyToSymbol(d_dimGrid, &dimGrid, sizeof(dimGrid));
  cout << "SP2D::setDimBlock: dimBlock " << dimBlock << " dimGrid " << dimGrid << endl;
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

void SP2D::setNumWall(long numWall_) {
  numWall = numWall_;
  cudaMemcpyToSymbol(d_numWall, &numWall, sizeof(numWall));
}

long SP2D::getNumWall() {
  long numWallFromDevice;
  cudaMemcpyFromSymbol(&numWallFromDevice, d_numWall, sizeof(d_numWall));
	return numWallFromDevice;
}

double SP2D::getWallRad() {
  double wallRadFromDevice;
  cudaMemcpyFromSymbol(&wallRadFromDevice, d_wallRad, sizeof(d_wallRad));
	return wallRadFromDevice;
}

double SP2D::getWallArea0() {
  double wallArea0FromDevice;
  cudaMemcpyFromSymbol(&wallArea0FromDevice, d_wallArea0, sizeof(d_wallArea0));
	return wallArea0FromDevice;
}

double SP2D::getWallArea() {
  double wallAreaFromDevice;
  cudaMemcpyFromSymbol(&wallAreaFromDevice, d_wallArea, sizeof(d_wallArea));
	return wallAreaFromDevice;
}

double SP2D::getWallAreaDeviation() {
  double wallAreaFromDevice;
  cudaMemcpyFromSymbol(&wallAreaFromDevice, d_wallArea, sizeof(d_wallArea));
  return (wallAreaFromDevice - wallArea0) / wallArea0;

}

std::tuple<double, double, double> SP2D::getWallAngleDynamics() {
  if(simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    return make_tuple(wallAngle, wallOmega, wallAlpha);
  } else {
    cout << "SP2D::getWallAngleDynamics only works for rigid boundary!" << endl;
    return make_tuple(0, 0, 0);
  }
}

void SP2D::setWallAngleDynamics(thrust::host_vector<double> wallDynamics_) {
  if(simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    wallAngle = wallDynamics_[0];
    wallOmega = wallDynamics_[1];
    wallAlpha = wallDynamics_[2];
    d_monomerAlpha.resize(numWall);
    thrust::fill(d_monomerAlpha.begin(), d_monomerAlpha.end(), double(0));
  } else {
    cout << "SP2D::getWallAngleDynamics only works for rigid boundary!" << endl;
  }
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

void SP2D::setBoxRadius(double boxRadius_) {
	syncSimControlFromDevice();
	if(simControl.geometryType == simControlStruct::geometryEnum::roundWall) {
		boxRadius = boxRadius_;
		cudaError err = cudaMemcpyToSymbol(d_boxRadius, &boxRadius, sizeof(boxRadius));
		if(err != cudaSuccess) {
			cout << "cudaMemcpyToSymbol Error: " << cudaGetErrorString(err) << endl;
		}
	}
	else {
		cout << "SP2D::setBoxRadius: attempting to set boxRadius without using round boundary conditions" << endl;
	}
  cout << "SP2D::setBoxRadius: boxRadius: " << boxRadius << endl;
}

void SP2D::scaleBoxRadius(double scale_) {
	syncSimControlFromDevice();
	if(simControl.geometryType == simControlStruct::geometryEnum::roundWall) {
		boxRadius = scale_ * boxRadius;
		cudaError err = cudaMemcpyToSymbol(d_boxRadius, &boxRadius, sizeof(boxRadius));
		if(err != cudaSuccess) {
			cout << "cudaMemcpyToSymbol Error: " << cudaGetErrorString(err) << endl;
		}
	}
	else {
		cout << "SP2D::scaleBoxRadius: attempting to scale boxRadius without using round boundary conditions" << endl;
	}
  cout << "SP2D::scaleBoxRadius: scale: " << scale_ << " boxRadius: " << boxRadius << endl;
}

double SP2D::getBoxRadius() {
  double boxRadiusFromDevice;
	cudaError err = cudaMemcpyFromSymbol(&boxRadiusFromDevice, d_boxRadius, sizeof(d_boxRadius));
	if(err != cudaSuccess) {
		cout << "cudaMemcpyToSymbol Error: " << cudaGetErrorString(err) << endl;
	}
	return boxRadiusFromDevice;
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
  return 2 * thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>()) / numParticles;
}

double SP2D::getMinParticleSigma() {
  return 2 * thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(1), thrust::minimum<double>());
}

double SP2D::getMaxParticleSigma() {
  return 2 * thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(-1), thrust::maximum<double>());
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

void SP2D::shrinkRadialCoordinates(double scale_) {
  // convert cartesian coordinates to polar coordinates
  // and multiply radial coordinate by a scalar factor
  if (simControl.geometryType == simControlStruct::geometryEnum::roundWall) {
    if(scale_ >= 1) {
      cout << "SP2D::shrinkRadialCoordinates: scale must be between 0 and 1!" << endl;
      return;
    } else {
      auto r = thrust::counting_iterator<long>(0);
      double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
      double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
      double *boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);

      auto scaleRadialCoordinate = [=] __device__ (long particleId) {
        double x = pPos[particleId * d_nDim];
        double y = pPos[particleId * d_nDim + 1];
        double radial = sqrt(x * x + y * y);
        if (radial > 0) {
          double theta = atan2(y, x);
          radial *= scale_;
          pPos[particleId * d_nDim] = radial * cos(theta);
          pPos[particleId * d_nDim + 1] = radial * sin(theta);
        }
      };

      thrust::for_each(r, r + numParticles, scaleRadialCoordinate);
      cout << "SP2D::shrinkRadialCoordinates: scale: " << scale_ << endl;
    }
  } else {
    cout << "SP2D::scaleRadialCoordinates: only works for roundWall geometry!" << endl;
  }
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

void SP2D::resetVicsekLastPositions() {
  cudaDeviceSynchronize();
  d_vicsekLastPos = d_particlePos;
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

thrust::host_vector<double> SP2D::getWallForces() {
  thrust::host_vector<double> wallForceFromDevice;
  wallForceFromDevice = d_wallForce;
  return wallForceFromDevice;
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

void SP2D::setWallPositions(thrust::host_vector<double> &wallPos_) {
  d_wallPos = wallPos_;
}

thrust::host_vector<double> SP2D::getWallPositions() {
  thrust::host_vector<double> wallPosFromDevice;
  wallPosFromDevice = d_wallPos;
  return wallPosFromDevice;
}

void SP2D::setWallVelocities(thrust::host_vector<double> &wallVel_) {
  d_wallVel = wallVel_;
}

thrust::host_vector<double> SP2D::getWallVelocities() {
  thrust::host_vector<double> wallVelFromDevice;
  wallVelFromDevice = d_wallVel;
  return wallVelFromDevice;
}

void SP2D::setWallLengths(thrust::host_vector<double> &wallLength_) {
  d_wallLength = wallLength_;
}

thrust::host_vector<double> SP2D::getWallLengths() {
  thrust::host_vector<double> wallLengthFromDevice;
  wallLengthFromDevice = d_wallLength;
  return wallLengthFromDevice;
}

void SP2D::setWallAngles(thrust::host_vector<double> &wallAngle_) {
  d_wallAngle = wallAngle_;
}

thrust::host_vector<double> SP2D::getWallAngles() {
  thrust::host_vector<double> wallAngleFromDevice;
  wallAngleFromDevice = d_wallAngle;
  return wallAngleFromDevice;
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
    switch (simControl.boundaryType) {
      case simControlStruct::boundaryEnum::mobile:
      case simControlStruct::boundaryEnum::plastic:
      if(wallArea > 0.) {
        return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / wallArea;
      } else {
        return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) / (boxRadius * boxRadius);
      }
      break;
      default:
      switch (simControl.geometryType) {
        case simControlStruct::geometryEnum::roundWall:
        return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) / (boxRadius * boxRadius);
        break;
        default:
        return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / (d_boxSize[0] * d_boxSize[1]);
        break;
      }
      break;
    }
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
  cutoff = cutoff_ * getMeanParticleSigma();
  cout << "SP2D::setDisplacementCutoff - cutDistance: " << cutDistance << " cutoff: " << cutoff << endl;
  return cutDistance;
}

// this function is called after particleDisplacement has been computed
void SP2D::removeCOMDrift() {
  getParticleMaxDisplacement();
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
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  int *flag = thrust::raw_pointer_cast(&d_flag[0]);
  kernelCheckParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, pLastPos, flag, cutoff);
  int sumFlag = thrust::reduce(d_flag.begin(), d_flag.end(), int(0), thrust::plus<int>());
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
    //checkParticleMaxDisplacement();
    checkParticleDisplacement();
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
    default:
    break;
  }
}

void SP2D::checkVicsekNeighbors() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *vLastPos = thrust::raw_pointer_cast(&d_vicsekLastPos[0]);
  int *vicsekFlag = thrust::raw_pointer_cast(&d_vicsekFlag[0]);
  kernelCheckParticleDisplacement<<<dimGrid,dimBlock>>>(pPos, vLastPos, vicsekFlag, Rvicsek);
  int sumFlag = thrust::reduce(d_vicsekFlag.begin(), d_vicsekFlag.end(), int(0), thrust::plus<int>());
  if(sumFlag != 0) {
    calcVicsekNeighborList();
    resetVicsekLastPositions();
  }
}

double SP2D::getSoftWaveNumber() {
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::mobile:
    case simControlStruct::boundaryEnum::plastic:
    return PI / (2. * sqrt(wallArea * getParticlePhi() / (PI * numParticles)));
    break;
    default:
    switch (simControl.geometryType) {
      case simControlStruct::geometryEnum::roundWall:
      if(nDim == 2) {
        return PI / (2. * sqrt(boxRadius * boxRadius * getParticlePhi() / numParticles));
      } else {
        cout << "SP2D::getSoftWaveNumber: only dimensions 2 in roundWall, rough, rigid, mobile and plastic geometry is allowed!" << endl;
        return 0;
      }
      break;
      default:
      if(nDim == 2) {
        return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getParticlePhi() / (PI * numParticles)));
      } else if (nDim == 3) {
        return PI / (2. * cbrt(d_boxSize[0] * d_boxSize[1] * d_boxSize[2] * getParticlePhi() / (PI * numParticles)));
      } else {
        cout << "SP2D::getSoftWaveNumber: only dimensions 2 and 3 are allowed!" << endl;
        return 0;
      }
      break;
    }
    break;
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
  setLengthScaleToOne();
}

// this only works in 2D
void SP2D::setRoundScaledPolyRandomParticles(double phi0, double polyDispersity, double boxRadius_) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean = 0, sigma, scale;
  double thisR, thisTheta;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = 0.5 * exp(mean + randNum * sigma);
  }
  boxRadius = boxRadius_;
  setBoxRadius(boxRadius);
  if(nDim == 2) {
  scale = sqrt(getParticlePhi() / phi0);
  } else {
    cout << "SP2D::setRoundScaledPolyRandomSoftParticles: only dimesions 2 is allowed!" << endl;
  }
  boxRadius = boxRadius_ * scale;
  setBoxRadius(boxRadius);
  // extract random positions
  for (long particleId = 0; particleId < numParticles; particleId++) {
    thisR = boxRadius * sqrt(drand48());
    thisTheta = 2 * PI  * drand48() - PI;
    d_particlePos[particleId * nDim] = thisR * cos(thisTheta);
    d_particlePos[particleId * nDim + 1] = thisR * sin(thisTheta);
  }
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
  double boxRadius_ = 0.;
  switch (simControl.geometryType) {
    case simControlStruct::geometryEnum::roundWall:
    boxRadius_ = getBoxRadius();
    boxRadius_ /= sigma;
    boxRadius = boxRadius_;
    cudaMemcpyToSymbol(d_boxRadius, &boxRadius, sizeof(boxRadius));
    break;
    default:
    thrust::host_vector<double> boxSize_(nDim);
    boxSize_ = getBoxSize();
    for (long dim = 0; dim < nDim; dim++) {
      boxSize_[dim] /= sigma;
    }
    d_boxSize = boxSize_;
    double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
    cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
    //setParticleLengthScale();
    break;
  }
}

void SP2D::scaleParticleVelocity(double scale) {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), thrust::make_constant_iterator(scale), d_particleVel.begin(), thrust::multiplies<double>());
}

// compute particle angles from velocity
void SP2D::initializeParticleAngles() {
  //thrust::counting_iterator<long> index_sequence_begin(lrand48());
  //thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleAngle.begin(), randNum(-PI, PI));
  long p_nDim(nDim);
  double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

  if(nDim == 2) {
    auto r = thrust::counting_iterator<long>(0);

    auto computeAngleFromVel2D = [=] __device__ (long particleId) {
      pAngle[particleId] = atan2(pVel[particleId * p_nDim + 1], pVel[particleId * p_nDim]);
    };

    thrust::for_each(r, r + numParticles, computeAngleFromVel2D);
  } 
  else if(nDim == 3) {
    auto s = thrust::counting_iterator<long>(0);

    auto computeAngleFromVel3D = [=] __device__ (long particleId) {
      auto theta = acos(pVel[particleId * p_nDim + 2]);
      auto phi = atan2(pVel[particleId * p_nDim + 1], pVel[particleId * p_nDim]);
      pAngle[particleId * p_nDim] = cos(theta) * cos(phi);
      pAngle[particleId * p_nDim + 1] = sin(theta) * cos(phi);
      pAngle[particleId * p_nDim + 2] = sin(phi);
    };

    thrust::for_each(s, s + numParticles, computeAngleFromVel3D);
  }
}

// define positions of monomers on wall by filling the circle of size 2 * PI * boxRadius
void SP2D::initRigidWall() {
  double circleLength = 2. * PI * boxRadius;
  numWall = circleLength / getMinParticleSigma();
  cudaMemcpyToSymbol(d_numWall, &numWall, sizeof(numWall));
  wallRad = 0.5 * (circleLength / numWall);
  cudaMemcpyToSymbol(d_wallRad, &wallRad, sizeof(wallRad));
  cout << "SP2D::initRigidWall:: wallRad: " << wallRad << " numWall: " << numWall << endl;
  initWallVariables(numWall);
  initWallNeighbors(numWall);
  for (long wallId = 0; wallId < numWall; wallId++) {
    d_wallPos[wallId * nDim] = boxRadius * cos((2. * PI * wallId) / numWall);
    d_wallPos[wallId * nDim + 1] = boxRadius * sin((2. * PI * wallId) / numWall);
  }
}

void SP2D::setWallShapeEnergyScales(double ea_, double el_, double eb_) {
  ea = ea_;
  el = el_;
  eb = eb_;
  cudaMemcpyToSymbol(d_ea, &ea, sizeof(ea));
  cudaMemcpyToSymbol(d_el, &el, sizeof(el));
  cudaMemcpyToSymbol(d_eb, &eb, sizeof(eb));
  cout << "SP2D::setWallShapeEnergyScales:: area: " << ea << " segment: " << el << " angle: " << eb << endl;
}

void SP2D::checkDimGrid() {
  if(numWall > numParticles) {
	  dimGrid = (numWall + dimBlock - 1) / dimBlock;
    cudaError err = cudaMemcpyToSymbol(d_dimGrid, &dimGrid, sizeof(dimGrid));
    cout << "SP2D::checkDimBlock: dimGrid " << dimGrid << endl;
    if(err != cudaSuccess) {
      cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
    }
  }
}

void SP2D::setRigidWallParams(long numWall_, double wallRad_) {
  numWall = numWall_;
  wallRad = wallRad_;
  cudaMemcpyToSymbol(d_numWall, &numWall, sizeof(numWall));
  cudaMemcpyToSymbol(d_wallRad, &wallRad, sizeof(wallRad));
  checkDimGrid();
}

void SP2D::setMobileWallParams(long numWall_, double wallRad_, double wallArea0_) {
  numWall = numWall_;
  wallRad = wallRad_;
  wallArea0 = wallArea0_;
  wallArea = wallArea0_;
  wallLength0 = 2 * wallRad_;
  wallAngle0 = 2. * PI / numWall_;
  cudaMemcpyToSymbol(d_numWall, &numWall, sizeof(numWall));
  cudaMemcpyToSymbol(d_wallRad, &wallRad, sizeof(wallRad));
  cudaMemcpyToSymbol(d_wallArea0, &wallArea0, sizeof(wallArea0));
  cudaMemcpyToSymbol(d_wallArea, &wallArea0, sizeof(wallArea0));
  cudaMemcpyToSymbol(d_wallLength0, &wallLength0, sizeof(wallLength0));
  cudaMemcpyToSymbol(d_wallAngle0, &wallAngle0, sizeof(wallAngle0));
  checkDimGrid();
}

void SP2D::setPlasticVariables(double lgamma_) {
  lgamma = lgamma_;
    thrust::fill(d_restLength.begin(), d_restLength.end(), wallLength0);
}

// define positions of monomers on wall by filling the circle of size 2 * PI * boxRadius
void SP2D::initMobileWall() {
  double circleLength = 2. * PI * boxRadius;
  numWall = circleLength / getMinParticleSigma();
  wallRad = 0.5 * (circleLength / numWall);
  cout << "SP2D::initMobileWall:: wallRad: " << wallRad << " numWall: " << numWall << endl;
  initWallVariables(numWall);
  initWallNeighbors(numWall);
  for (long wallId = 0; wallId < numWall; wallId++) {
    d_wallPos[wallId * nDim] = boxRadius * cos((2. * PI * wallId) / numWall);
    d_wallPos[wallId * nDim + 1] = boxRadius * sin((2. * PI * wallId) / numWall);
  }
  wallArea0 = PI * boxRadius * boxRadius;
  setMobileWallParams(numWall, wallRad, wallArea0);
  if(simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    setPlasticVariables(1.);
  }
  initWallShapeVariables(numWall);
  // increase boxRadius to track movement of boundary
  scaleBoxRadius(1.1);
}

void SP2D::initializeWall() {
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::rough:
    case simControlStruct::boundaryEnum::rigid:
    initRigidWall();
    break;
    case simControlStruct::boundaryEnum::mobile:
    case simControlStruct::boundaryEnum::plastic:
    initMobileWall();
    break;
    default:
    break;
  }
}

//*************************** force and energy *******************************//
void SP2D::setEnergyCostant(double ec_) {
  ec = ec_;
  cudaMemcpyToSymbol(d_ec, &ec, sizeof(ec));
  setWallEnergyScale(ec);
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

void SP2D::setVicsekParams(double driving_, double taup_, double Jvicsek_, double Rvicsek_) {
  driving = driving_;
  taup = taup_;
  Jvicsek = Jvicsek_;
  Rvicsek = Rvicsek_;
  double boxRadius = getBoxRadius();
  if(Rvicsek > 0.5 * boxRadius) {
    Rvicsek = 0.5 * boxRadius;
    cout << "SP2D::setVicsekParams:: Rvicsek cannot be grater than half the boxRadius, setting Rvicsek equal to half the boxRadius" << endl;
  }
  cudaMemcpyToSymbol(d_driving, &driving, sizeof(driving));
  cudaMemcpyToSymbol(d_taup, &taup, sizeof(taup));
  cudaMemcpyToSymbol(d_Jvicsek, &Jvicsek, sizeof(Jvicsek));
  //cout << "SP2D::setVicsekParams:: driving: " << driving << " interactin strength: " << Jvicsek << " and radius: " << Rvicsek << endl;
}

void SP2D::getVicsekParams(double &driving_, double &taup_, double &Jvicsek_, double &Rvicsek_) {
  driving_ = driving;
  taup_ = taup;
  Jvicsek_ = Jvicsek;
  Rvicsek_ = Rvicsek;
  //cout << "SP2D::getVicsekParams:: driving: " << driving_ << " interactin strength: " << Jvicsek << " and radius: " << Rvicsek << endl;
}

void SP2D::setReflectionNoise(double angleAmplitude_) {
  angleAmplitude = angleAmplitude_;
  //cout << "SP2D::setReflectionNoise:: angleAmplitude: " << angleAmplitude << endl;
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
  d_flagAB.resize(numParticles);
  thrust::fill(d_flagAB.begin(), d_flagAB.end(), 0);
  d_squaredVelAB.resize(numParticles * nDim);
  d_particleEnergyAB.resize(numParticles);
  thrust::fill(d_squaredVelAB.begin(), d_squaredVelAB.end(), double(0));
  thrust::fill(d_particleEnergyAB.begin(), d_particleEnergyAB.end(), double(0));
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  eAA = eAA_;
  eAB = eAB_;
  eBB = eBB_;
  cudaMemcpyToSymbol(d_eAA, &eAA, sizeof(eAA));
  cudaMemcpyToSymbol(d_eAB, &eAB, sizeof(eAB));
  cudaMemcpyToSymbol(d_eBB, &eBB, sizeof(eBB));
  setWallEnergyScale(eAA);
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

void SP2D::setWallEnergyScale(double ew_) {
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
  // use this to be able to set taup to 0
  double amplitude = 0.;
  if(taup != 0) {
    amplitude = sqrt(2.0 * dt / taup);
  }
  auto r = thrust::counting_iterator<long>(0);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	if(nDim == 2) {
    thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_randAngle.begin(), wrappedGaussNum(0.f,amplitude));
    double *randAngle = thrust::raw_pointer_cast(&d_randAngle[0]);

    auto updateActiveNoise2D = [=] __device__ (long pId) {
      pAngle[pId] += randAngle[pId];
      pAngle[pId] = pAngle[pId] + PI;
      pAngle[pId] = pAngle[pId] - 2.0 * PI * floor(pAngle[pId] / (2.0 * PI));
      pAngle[pId] = pAngle[pId] - PI;
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pForce[pId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId]));
      }
    };

    thrust::for_each(r, r + numParticles, updateActiveNoise2D);

  } else if(nDim == 3) {
    auto s = thrust::counting_iterator<long>(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + numParticles * nDim, d_randAngle.begin(), gaussNum(0.f,1.f));
    double *randAngle = thrust::raw_pointer_cast(&d_randAngle[0]);

    auto normalizeVector = [=] __device__ (long particleId) {
      auto norm = 0.0;
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < nDim; dim++) {
        norm += randAngle[particleId * s_nDim + dim] * randAngle[particleId * s_nDim + dim];
      }
      norm = sqrt(norm);
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        randAngle[particleId * s_nDim + dim] /= norm;
      }
    };

    thrust::for_each(s, s + numParticles, normalizeVector);

    auto updateActiveNoise3D = [=] __device__ (long particleId) {
      pAngle[particleId * s_nDim] += amplitude * (pAngle[particleId * s_nDim + 1] * randAngle[particleId * s_nDim + 2] - pAngle[particleId * s_nDim + 2] * randAngle[particleId * s_nDim + 1]);
      pAngle[particleId * s_nDim + 1] += amplitude * (pAngle[particleId * s_nDim + 2] * randAngle[particleId * s_nDim] - pAngle[particleId * s_nDim] * randAngle[particleId * s_nDim + 2]);
      pAngle[particleId * s_nDim + 2] += amplitude * (pAngle[particleId * s_nDim] * randAngle[particleId * s_nDim + 1] - pAngle[particleId * s_nDim + 1] * randAngle[particleId * s_nDim]);
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pForce[particleId * s_nDim + dim] += s_driving * pAngle[particleId * s_nDim + dim];
      }
    };

    thrust::for_each(r, r + numParticles, updateActiveNoise3D);
  }
}

void SP2D::addVicsekAlignment() {
	if(nDim == 2) {
    int s_nDim(nDim);
    double s_dt(dt);
    double s_driving(driving);
    // use this to be able to set taup to 0
    double amplitude = 0.;
    if(taup != 0) {
      amplitude = sqrt(2.0 * dt / taup);
    }
    auto r = thrust::counting_iterator<long>(0);
    thrust::counting_iterator<long> index_sequence_begin(lrand48());
    thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_randAngle.begin(), wrappedGaussNum(0.f,amplitude));
    const double *pAlpha = thrust::raw_pointer_cast(&d_particleAlpha[0]);
    double *randAngle = thrust::raw_pointer_cast(&d_randAngle[0]);
    double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
    double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

    auto updateVicsekAlignment2D = [=] __device__ (long pId) {
      // overdamped equation for the angle with vicsek alignment as torque
      pAngle[pId] += randAngle[pId] + s_dt * pAlpha[pId];
      pAngle[pId] = pAngle[pId] + PI;
      pAngle[pId] = pAngle[pId] - 2.0 * PI * floor(pAngle[pId] / (2.0 * PI));
      pAngle[pId] = pAngle[pId] - PI;
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        pForce[pId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId]));
      }
    };

    thrust::for_each(r, r + numParticles, updateVicsekAlignment2D);
  }
}

void SP2D::calcVicsekAlignment() {
  checkVicsekNeighbors();
  const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  double *pAlpha = thrust::raw_pointer_cast(&d_particleAlpha[0]);
  switch (simControl.alignType) {
    case simControlStruct::alignEnum::additive:
    kernelCalcVicsekAdditiveAlignment<<<dimGrid, dimBlock>>>(pAngle, pAlpha);
    break;
    case simControlStruct::alignEnum::nonAdditive:
    kernelCalcVicsekNonAdditiveAlignment<<<dimGrid, dimBlock>>>(pAngle, pAlpha);
    break;
    case simControlStruct::alignEnum::velAlign:
    const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
    kernelCalcVicsekVelocityAlignment<<<dimGrid, dimBlock>>>(pVel, pAlpha);
    break;
  }
}

void SP2D::calcParticleFixedWallInteraction() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  switch (simControl.geometryType) {
    case simControlStruct::geometryEnum::squareWall:
    kernelCalcParticleSquareWallInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wForce);
    break;
    case simControlStruct::geometryEnum::fixedSides2D:
    kernelCalcParticleSidesInteraction2D<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wForce);
    break;
    case simControlStruct::geometryEnum::fixedSides3D:
    kernelCalcParticleSidesInteraction3D<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wForce);
    break;
    case simControlStruct::geometryEnum::roundWall:
    kernelCalcParticleRoundWallInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wForce);
    break;
    default:
    break;
  }
}

void SP2D::calcWallArea() {
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
  double *aSector = thrust::raw_pointer_cast(&d_areaSector[0]);
  kernelCalcWallArea<<<dimGrid, dimBlock>>>(wPos, aSector);
  wallArea = 0.5 * fabs(thrust::reduce(d_areaSector.begin(), d_areaSector.end(), double(0), plus<double>()));
  cudaMemcpyToSymbol(d_wallArea, &wallArea, sizeof(wallArea));
  //cout << "SP2D::calcWallArea: wallArea " << wallArea << endl;
}

void SP2D::calcWallShape() {
  calcWallArea();
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
  double *wLength = thrust::raw_pointer_cast(&d_wallLength[0]);
  double *wAngle = thrust::raw_pointer_cast(&d_wallAngle[0]);
  kernelCalcWallShape<<<dimGrid, dimBlock>>>(wPos, wLength, wAngle);
}

void SP2D::calcWallShapeForceEnergy() {
  calcWallShape();
  const double *wLength = thrust::raw_pointer_cast(&d_wallLength[0]);
  const double *wAngle = thrust::raw_pointer_cast(&d_wallAngle[0]);
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
	double *wEnergy = thrust::raw_pointer_cast(&d_wallEnergy[0]);
  kernelCalcWallShapeForceEnergy<<<dimGrid, dimBlock>>>(wLength, wAngle, wPos, wForce, wEnergy);
}

void SP2D::calcPlasticWallShapeForceEnergy() {
  calcWallShape();
  const double *wLength = thrust::raw_pointer_cast(&d_wallLength[0]);
  const double *rLength = thrust::raw_pointer_cast(&d_restLength[0]);
  const double *wAngle = thrust::raw_pointer_cast(&d_wallAngle[0]);
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
	double *wEnergy = thrust::raw_pointer_cast(&d_wallEnergy[0]);
  kernelCalcPlasticWallShapeForceEnergy<<<dimGrid, dimBlock>>>(wLength, rLength, wAngle, wPos, wForce, wEnergy);
}

void SP2D::calcParticleWallInteraction() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // wall variables
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
	double *wEnergy = thrust::raw_pointer_cast(&d_wallEnergy[0]);
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    kernelCalcParticleWallInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wPos, wForce, wEnergy);
    //kernelCalcSmoothWallInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wPos, wForce, wEnergy);
    break;
    case simControlStruct::neighborEnum::allToAll:
    kernelCalcAllToWallParticleInteraction<<<dimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy, wPos, wForce, wEnergy);
    break;
  }
}

void SP2D::calcWallAngularAcceleration() {
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
	double *mAlpha = thrust::raw_pointer_cast(&d_monomerAlpha[0]);
  kernelCalcWallAngularAcceleration<<<dimGrid, dimBlock>>>(wPos, wForce, mAlpha);
  wallAlpha = thrust::reduce(d_monomerAlpha.begin(), d_monomerAlpha.end(), double(0), plus<double>());
}

void SP2D::addParticleGravity() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  kernelAddParticleGravity<<<dimGrid, dimBlock>>>(pPos, pForce, pEnergy);
}

void SP2D::calcParticleForceEnergy() {
  switch (simControl.potentialType) {
    case simControlStruct::potentialEnum::none:
    break;
    default:
    calcParticleInteraction();
    break;
  }
  switch (simControl.particleType) {
    case simControlStruct::particleEnum::active:
    addSelfPropulsion();
    break;
    case simControlStruct::particleEnum::vicsek:
    calcVicsekAlignment();
    addVicsekAlignment();
    break;
    default:
    break;
  }
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::fixed:
    calcParticleFixedWallInteraction();
    break;
    case simControlStruct::boundaryEnum::rough:
    // Setting wallForce and wallEnergy to zero here and in the next case 
    // is necessary because calcParticleWallInteraction does not reset them
    // since it is used in combination with the deformable wall force functions
    // which do reset the wall forces and energies.
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    thrust::fill(d_wallEnergy.begin(), d_wallEnergy.end(), double(0));
    calcParticleWallInteraction();
    break;
    case simControlStruct::boundaryEnum::rigid:
    thrust::fill(d_wallForce.begin(), d_wallForce.end(), double(0));
    thrust::fill(d_wallEnergy.begin(), d_wallEnergy.end(), double(0));
    calcParticleWallInteraction();
    // transform particle-wall interaction into angular acceleration
    calcWallAngularAcceleration();
    break;
    case simControlStruct::boundaryEnum::mobile:
    calcWallShapeForceEnergy();
    calcParticleWallInteraction();
    break;
    case simControlStruct::boundaryEnum::plastic:
    calcPlasticWallShapeForceEnergy();
    calcParticleWallInteraction();
    break;
    default:
    break;
  }
  switch (simControl.gravityType) {
    case simControlStruct::gravityEnum::on:
    addParticleGravity();
    break;
    default:
    break;
  }
}

void SP2D::checkParticleInsideRoundWall() {
  double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  kernelCheckParticleInsideRoundWall<<<dimGrid, dimBlock>>>(pPos);
}

void SP2D::checkReflectiveWall() {
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::reflect:
    reflectParticleOnWall();
    break;
    case simControlStruct::boundaryEnum::reflectNoise:
    reflectParticleOnWallWithNoise();
    break;
    default:
    break;
  }
}

void SP2D::reflectParticleOnWall() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  switch (simControl.geometryType) {
    case simControlStruct::geometryEnum::squareWall:
    kernelReflectParticleFixedWall<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, wForce);
    break;
    case simControlStruct::geometryEnum::fixedSides2D:
    kernelReflectParticleFixedSides2D<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, wForce);
    break;
    case simControlStruct::geometryEnum::fixedSides3D:
    kernelReflectParticleFixedSides3D<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, wForce);
    break;
		case simControlStruct::geometryEnum::roundWall:
    kernelReflectParticleRoundWall<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, pAngle, wForce);
    break;
    default:
    break;
	}
}

void SP2D::reflectParticleOnWallWithNoise() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_randomAngle.begin(), wrappedGaussNum(0.f,1.f));
  thrust::transform(d_randomAngle.begin(), d_randomAngle.end(), thrust::make_constant_iterator(angleAmplitude), d_randomAngle.begin(), thrust::multiplies<double>());
  const double *randAngle = thrust::raw_pointer_cast(&d_randomAngle[0]);
  switch (simControl.geometryType) {
		case simControlStruct::geometryEnum::squareWall:
    kernelReflectParticleFixedWallWithNoise<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, randAngle, wForce);
    break;
		case simControlStruct::geometryEnum::roundWall:
    kernelReflectParticleRoundWallWithNoise<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, pAngle, randAngle, wForce);
    break;
    default:
    break;
	}
}

std::tuple<double, double, double> SP2D::getVicsekOrderParameters() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *unitPos = thrust::raw_pointer_cast(&d_unitPos[0]);
  double *unitVel = thrust::raw_pointer_cast(&d_unitVel[0]);
  double *unitVelPos = thrust::raw_pointer_cast(&d_unitVelPos[0]);
  kernelCalcUnitPosVel<<<dimGrid, dimBlock>>>(pPos, pVel, unitPos, unitVel, unitVelPos);
  // compute phase order parameters
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> unitPos_re(d_unitPos.begin(), d_unitPos.end(), 2);
  strided_range<Iterator> unitPos_im(d_unitPos.begin() + 1, d_unitPos.end(), 2);
  double realField = thrust::reduce(unitPos_re.begin(), unitPos_re.end(), double(0), thrust::plus<double>()) / numParticles;
  double imagField = thrust::reduce(unitPos_im.begin(), unitPos_im.end(), double(0), thrust::plus<double>()) / numParticles;
  double param1 = sqrt(realField * realField + imagField * imagField);
  // compute velocity order parameters
  strided_range<Iterator> unitVel_re(d_unitVel.begin(), d_unitVel.end(), 2);
  strided_range<Iterator> unitVel_im(d_unitVel.begin() + 1, d_unitVel.end(), 2);
  realField = thrust::reduce(unitVel_re.begin(), unitVel_re.end(), double(0), thrust::plus<double>()) / numParticles;
  imagField = thrust::reduce(unitVel_im.begin(), unitVel_im.end(), double(0), thrust::plus<double>()) / numParticles;
  double param2 = sqrt(realField * realField + imagField * imagField);
  // compute velocity-position order parameters
  strided_range<Iterator> unitVelPos_re(d_unitVelPos.begin(), d_unitVelPos.end(), 2);
  strided_range<Iterator> unitVelPos_im(d_unitVelPos.begin() + 1, d_unitVelPos.end(), 2);
  realField = thrust::reduce(unitVelPos_re.begin(), unitVelPos_re.end(), double(0), thrust::plus<double>()) / numParticles;
  imagField = thrust::reduce(unitVelPos_im.begin(), unitVelPos_im.end(), double(0), thrust::plus<double>()) / numParticles;
  double param3 = sqrt(realField * realField + imagField * imagField);
  return std::make_tuple(param1, param2, param3);
}

double SP2D::getVicsekHigherOrderParameter(double order_) {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *unitPos = thrust::raw_pointer_cast(&d_unitPos[0]);
  kernelCalcHigherOrderUnitVel<<<dimGrid, dimBlock>>>(pPos, unitPos, order_);
  // compute phase order parameters
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> unitPos_re(d_unitPos.begin(), d_unitPos.end(), 2);
  strided_range<Iterator> unitPos_im(d_unitPos.begin() + 1, d_unitPos.end(), 2);
  double realField = thrust::reduce(unitPos_re.begin(), unitPos_re.end(), double(0), thrust::plus<double>()) / numParticles;
  double imagField = thrust::reduce(unitPos_im.begin(), unitPos_im.end(), double(0), thrust::plus<double>()) / numParticles;
  return sqrt(realField * realField + imagField * imagField);
}

double SP2D::getVicsekVelocityCorrelation() {
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *velCorr = thrust::raw_pointer_cast(&d_velCorr[0]);
  kernelCalcVicsekVelocityCorrelation<<<dimGrid, dimBlock>>>(pVel, velCorr);
  return thrust::reduce(d_velCorr.begin(), d_velCorr.end(), double(0), thrust::plus<double>()) / numParticles;
}

double SP2D::getNeighborVelocityCorrelation() {
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *velCorr = thrust::raw_pointer_cast(&d_velCorr[0]);
  kernelCalcNeighborVelocityCorrelation<<<dimGrid, dimBlock>>>(pVel, velCorr);
  return thrust::reduce(d_velCorr.begin(), d_velCorr.end(), double(0), thrust::plus<double>()) / numParticles;
}

double SP2D::getParticleAngularMomentum() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double *angMom = thrust::raw_pointer_cast(&d_angMom[0]);
  kernelCalcParticleAngularMomentum<<<dimGrid, dimBlock>>>(pPos, pVel, angMom);
  return thrust::reduce(d_angMom.begin(), d_angMom.end(), double(0), thrust::plus<double>()) / numParticles;
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
  double *pStress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcStressTensor<<<dimGrid, dimBlock>>>(pRad, pPos, pStress);
}

void SP2D::define2DStressGrid(double binSize_) {
  binSize = binSize_ * getMeanParticleSigma();
  cudaMemcpyToSymbol(d_binSize, &binSize, sizeof(binSize));

  nBinsX = long(d_boxSize[0] / binSize);
  nBinsY = long(d_boxSize[1] / binSize);
  cudaMemcpyToSymbol(d_nBinsX, &nBinsX, sizeof(nBinsX));
  cudaMemcpyToSymbol(d_nBinsY, &nBinsY, sizeof(nBinsY));
  // 2 components per bin: xx and yy
  gridSize = nBinsX * nBinsY * 2;

  d_kinStress.resize(gridSize);
  d_confStress.resize(gridSize);
  cout << "SP2D::define2DStressGrid: nBinsX: " << nBinsX << " nBinsY: " << nBinsY << " gridSize: " << gridSize << endl;
}

void SP2D::calc2DStressProfile() {
  thrust::fill(d_kinStress.begin(), d_kinStress.end(), double(0));
  thrust::fill(d_confStress.begin(), d_confStress.end(), double(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  double* pKinStress = thrust::raw_pointer_cast(&d_kinStress[0]);
  double* pConfStress = thrust::raw_pointer_cast(&d_confStress[0]);
  kernelCalc2DStressProfile<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, pKinStress, pConfStress);
}

thrust::host_vector<double> SP2D::get2DStressProfile() {
  calc2DStressProfile();
  // average over vertical grid
  thrust::host_vector<double> kinStress = d_kinStress;
  thrust::host_vector<double> confStress = d_confStress;

  std::vector<double> avgKinStressX(nBinsX * 2, 0.0); // 2 is for xx and yy components
  std::vector<double> avgConfStressX(nBinsX * 2, 0.0); // 2 is for xx and yy components

  for (long x = 0; x < nBinsX; ++x) {
    for (long y = 0; y < nBinsY; ++y) {
        long idx = 2 * (y * nBinsX + x);
        avgKinStressX[2 * x] += kinStress[idx]; // xx
        avgKinStressX[2 * x + 1] += kinStress[idx + 1]; // yy
        avgConfStressX[2 * x] += confStress[idx]; // xx
        avgConfStressX[2 * x + 1] += confStress[idx + 1]; // yy
      }
    avgKinStressX[2 * x] /= nBinsY;
    avgKinStressX[2 * x + 1] /= nBinsY;
    avgConfStressX[2 * x] /= nBinsY;
    avgConfStressX[2 * x + 1] /= nBinsY;
  }
  // Create 5-column host vector: [x_bin_center, kin_xx, kin_yy, conf_xx, conf_yy]
  thrust::host_vector<double> stressProfile(nBinsX * 5);
  for (long x = 0; x < nBinsX; ++x) {
    double binCenter = (x + 0.5) * binSize;
    stressProfile[5 * x]     = binCenter;
    stressProfile[5 * x + 1] = avgKinStressX[2 * x];       // kinetic xx
    stressProfile[5 * x + 2] = avgKinStressX[2 * x + 1];   // kinetic yy
    stressProfile[5 * x + 3] = avgConfStressX[2 * x];      // config xx
    stressProfile[5 * x + 4] = avgConfStressX[2 * x + 1];  // config yy
  }
  return stressProfile;
}

double SP2D::getParticlePressure() {
  calcParticleStressTensor();
  double volume = 1.0;
  double stress = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
    stress += d_stress[dim * nDim + dim];
  }
  stress /= nDim;
  // add kinetic stress
  stress += getParticleKineticEnergy();
  stress /= volume;
  return stress;
}

double SP2D::getParticleTotalPressure() {
  calcParticleStressTensor();
  double volume = 1.0;
  double stress = 0.0;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
    stress += d_stress[dim * nDim + dim];
  }
  // add non conservative stress
  stress += getParticleWork();
  stress /= nDim;
  // add kinetic stress
  stress += getParticleKineticEnergy();
  stress /= volume;
  return stress;
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

// TODO: add roundWall geometry
std::tuple<double, double> SP2D::computeWallPressure() {
  thrust::device_vector<double> d_wallStress(d_particleForce.size());
  thrust::fill(d_wallStress.begin(), d_wallStress.end(), double(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wallStress = thrust::raw_pointer_cast(&d_wallStress[0]);
  double boxLength = 0.;
  switch (simControl.geometryType) {
		case simControlStruct::geometryEnum::squareWall:
    kernelCalcWallStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
    boxLength = 2. * thrust::reduce(d_boxSize.begin(), d_boxSize.end(), double(0), thrust::plus<double>());
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
    kernelCalcSides2DStress<<<dimGrid, dimBlock>>>(pRad, pPos, wallStress);
    boxLength = 2. * d_boxSize[1];
    break;
    default:
    break;
	}
  if(boxLength != 0.) {
    typedef thrust::device_vector<double>::iterator Iterator;
    strided_range<Iterator> xWallStress(d_wallStress.begin(), d_wallStress.end(), 2);
    strided_range<Iterator> yWallStress(d_wallStress.begin() + 1, d_wallStress.end(), 2);
    double xPressure = thrust::reduce(xWallStress.begin(), xWallStress.end(), double(0), thrust::plus<double>()) / boxLength;
    double yPressure = thrust::reduce(yWallStress.begin(), yWallStress.end(), double(0), thrust::plus<double>()) / boxLength;
    return std::make_tuple(xPressure, yPressure);
  } else {
    cout << "SP2D::computeWallPressure: Warning! boxLength is zero!" << endl;
    return std::make_tuple(0, 0);
  }
}

void SP2D::convertFixedWallForceToRadial() {
  // convert wallForce from cartesian to polar coordinates
  auto r = thrust::counting_iterator<long>(0);
  double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);

  // wallForce in fixed wall boundary is defined in the particle coordinate system
  auto convertFixedWallForce = [=] __device__ (long particleId) {
    if(wForce[particleId * d_nDim] != 0. && wForce[particleId * d_nDim + 1] != 0.) {
      double x = pPos[particleId * d_nDim];
      double y = pPos[particleId * d_nDim + 1];
      double theta = atan2(y, x);
      double force_rad = cos(theta) * wForce[particleId * d_nDim] + sin(theta) * wForce[particleId * d_nDim + 1];
      double force_tan = -sin(theta) * wForce[particleId * d_nDim] + cos(theta) * wForce[particleId * d_nDim + 1];
      wForce[particleId * d_nDim] = force_rad;
      wForce[particleId * d_nDim + 1] = force_tan;
    }
  };

  thrust::for_each(r, r + numParticles, convertFixedWallForce);
}

void SP2D::convertRoughWallForceToRadial() {
  // convert wallForce from cartesian to polar coordinates
  auto r = thrust::counting_iterator<long>(0);
  double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
  double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);

  auto convertRoughWallForce = [=] __device__ (long wallId) {
    double x = wPos[wallId * d_nDim];
    double y = wPos[wallId * d_nDim + 1];
    double theta = atan2(y, x);
    double force_rad = cos(theta) * wForce[wallId * d_nDim] + sin(theta) * wForce[wallId * d_nDim + 1];
    double force_tan = -sin(theta) * wForce[wallId * d_nDim] + cos(theta) * wForce[wallId * d_nDim + 1];
    wForce[wallId * d_nDim] = force_rad;
    wForce[wallId * d_nDim + 1] = force_tan;
  };

  thrust::for_each(r, r + numWall, convertRoughWallForce);
}

void SP2D::convertMobileWallForceToRadial() {
  // first compute wall center of mass
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> xWallPos(d_wallPos.begin(), d_wallPos.end(), 2);
  strided_range<Iterator> yWallPos(d_wallPos.begin() + 1, d_wallPos.end(), 2);
  double xWall = thrust::reduce(xWallPos.begin(), xWallPos.end(), double(0), thrust::plus<double>()) / numWall;
  double yWall = thrust::reduce(yWallPos.begin(), yWallPos.end(), double(0), thrust::plus<double>()) / numWall;

  // convert wallForce from cartesian to polar coordinates
  auto r = thrust::counting_iterator<long>(0);
  double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);
  double *wForce = thrust::raw_pointer_cast(&d_wallForce[0]);

  auto convertMobileWallForce = [=] __device__ (long wallId) {
    double x = wPos[wallId * d_nDim] - xWall; // assumes boxSize is much bigger than wall size
    double y = wPos[wallId * d_nDim + 1] - yWall;
    double theta = atan2(y, x);
    double force_rad = cos(theta) * wForce[wallId * d_nDim] + sin(theta) * wForce[wallId * d_nDim + 1];
    double force_tan = -sin(theta) * wForce[wallId * d_nDim] + cos(theta) * wForce[wallId * d_nDim + 1];
    wForce[wallId * d_nDim] = force_rad;
    wForce[wallId * d_nDim + 1] = force_tan;
  };

  thrust::for_each(r, r + numWall, convertMobileWallForce);
}

std::tuple<double, double> SP2D::getWallPressure() {
  if(nDim == 2) {
    double boxLength = 0.;
    switch (simControl.geometryType) {
      case simControlStruct::geometryEnum::roundWall:
      boxLength = 2. * PI * boxRadius;
      switch (simControl.boundaryType) {
        case simControlStruct::boundaryEnum::fixed:
        convertFixedWallForceToRadial();
        break;
        case simControlStruct::boundaryEnum::rough:
        case simControlStruct::boundaryEnum::rigid:
        convertRoughWallForceToRadial();
        break;
        case simControlStruct::boundaryEnum::mobile:
        case simControlStruct::boundaryEnum::plastic:
        convertMobileWallForceToRadial();
        boxLength = thrust::reduce(d_wallLength.begin(), d_wallLength.end(), double(0), thrust::plus<double>());
        default:
        break;
      }
      break;
      default:
      boxLength = 2. * thrust::reduce(d_boxSize.begin(), d_boxSize.end(), double(0), thrust::plus<double>());
      break;
    }
    if(boxLength != 0.) {
      typedef thrust::device_vector<double>::iterator Iterator;
      strided_range<Iterator> xWallForce(d_wallForce.begin(), d_wallForce.end(), 2);
      strided_range<Iterator> yWallForce(d_wallForce.begin() + 1, d_wallForce.end(), 2);
      double xForce = thrust::reduce(xWallForce.begin(), xWallForce.end(), double(0), thrust::plus<double>()) / boxLength;
      double yForce = thrust::reduce(yWallForce.begin(), yWallForce.end(), double(0), thrust::plus<double>()) / boxLength;
      return std::make_tuple(xForce, yForce);
    } else {
      cout << "SP2D::getWallPressure: Warning! boxLength is zero!" << endl;
      return std::make_tuple(0, 0);
    }
  } else {
    cout << "SP2D::getWallPressure: WORK IN PROGRESS, only works for 2D" << endl;
    return std::make_tuple(0, 0);
  }
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

double SP2D::getWallPotentialEnergy() {
  return thrust::reduce(d_wallEnergy.begin(), d_wallEnergy.end(), double(0), thrust::plus<double>());
}

double SP2D::getParticleKineticEnergy() {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), d_squaredVel.begin(), square());
  return 0.5 * thrust::reduce(d_squaredVel.begin(), d_squaredVel.end(), double(0), thrust::plus<double>());
}

double SP2D::getWallKineticEnergy() {
  thrust::transform(d_wallVel.begin(), d_wallVel.end(), d_sqWallVel.begin(), square());
  return 0.5 * thrust::reduce(d_sqWallVel.begin(), d_sqWallVel.end(), double(0), thrust::plus<double>());
}

double SP2D::getWallRotationalKineticEnergy() {
  return 0.5 * boxRadius * boxRadius * wallOmega * wallOmega;
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

double SP2D::getNoiseWork() {
  thrust::device_vector<double> d_noiseWork(d_particleEnergy.size());
  thrust::fill(d_noiseWork.begin(), d_noiseWork.end(), double(0));

  double s_dt(dt);
  long s_nDim(nDim);
  double s_noise(this->sim_->noise);
  switch (simControl.noiseType) {
    case simControlStruct::noiseEnum::langevin2:
    s_noise *= 0.5; // a factor of 2 comes out of the integration scheme for the second-order Langevin integrator
    break;
    default:
    break;
  }
  auto r = thrust::counting_iterator<long>(0);
	const double* rand = thrust::raw_pointer_cast(&(this->sim_->d_rand[0]));
	const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double* nWork = thrust::raw_pointer_cast(&d_noiseWork[0]);

  auto computeNoiseWork = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      nWork[pId] += s_dt * s_noise * rand[pId * s_nDim + dim] * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + numParticles, computeNoiseWork);

  return thrust::reduce(d_noiseWork.begin(), d_noiseWork.end(), double(0), thrust::plus<double>());
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

double SP2D::getParticleWork() {
  double work = getDampingWork() + getNoiseWork();
  if(simControl.particleType == simControlStruct::particleEnum::active) {
    work += getSelfPropulsionWork();
  }
  return work;
}

double SP2D::getParticleTemperature() {
  return 2 * getParticleKineticEnergy() / (nDim * numParticles);
}

double SP2D::getParticleEnergy() {
  return (getParticlePotentialEnergy() + getParticleKineticEnergy());
}

double SP2D::getWallTemperature() {
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::mobile:
    case simControlStruct::boundaryEnum::plastic:
    return 2 * getWallKineticEnergy() / (nDim * numWall);
    break;
    case simControlStruct::boundaryEnum::rigid:
    return 2 * getWallRotationalKineticEnergy() / nDim;
    break;
    default:
    return 0;
    break;
  }
}

double SP2D::getWallEnergy() {
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::fixed:
    case simControlStruct::boundaryEnum::rough:
    return getWallPotentialEnergy();
    break;
    case simControlStruct::boundaryEnum::rigid:
    return getWallPotentialEnergy() + getWallRotationalKineticEnergy();
    break;
    case simControlStruct::boundaryEnum::mobile:
    case simControlStruct::boundaryEnum::plastic:
    return getWallPotentialEnergy() + getWallKineticEnergy();
    break;
    default:
    return 0;
    break;
  }
}

double SP2D::getTotalEnergy() {
  return getWallEnergy() + getParticleEnergy();
}

std::tuple<double, double, double> SP2D::getParticleKineticEnergy12() {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), d_squaredVel.begin(), square());
  thrust::device_vector<double> velSq1(num1 * nDim);
  thrust::device_vector<double> velSq2((numParticles-num1) * nDim);
  thrust::copy(d_squaredVel.begin(), d_squaredVel.begin() + num1 * nDim, velSq1.begin());
  thrust::copy(d_squaredVel.begin() + num1 * nDim, d_squaredVel.end(), velSq2.begin());
  double ekin1 = 0.5 * thrust::reduce(velSq1.begin(), velSq1.end(), double(0), thrust::plus<double>());
  double ekin2 = 0.5 * thrust::reduce(velSq2.begin(), velSq2.end(), double(0), thrust::plus<double>());
  double ekin = 0.5 * thrust::reduce(d_squaredVel.begin(), d_squaredVel.end(), double(0), thrust::plus<double>());
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

void SP2D::adjustLocalKineticEnergy(thrust::host_vector<double> &prevEnergy_, long direction_) {
  thrust::device_vector<double> d_prevEnergy = prevEnergy_;
  // compute new potential energy per particle
  getParticlePotentialEnergy();
  // label particles close to compressing walls
  thrust::device_vector<long> d_wallLabel(numParticles);
  thrust::fill(d_wallLabel.begin(), d_wallLabel.end(), 0);
  long *wallLabel = thrust::raw_pointer_cast(&d_wallLabel[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  kernelAssignWallLabel<<<dimGrid,dimBlock>>>(pPos, pRad, wallLabel, direction_);

  /*for (long pId=0; pId<numParticles; pId++) {
    if(d_wallLabel[pId] == 1) {
      if(direction_ == 1) {
        cout << "Particle " << pId << " near vertical wall, pos: " << d_particlePos[pId * nDim] << endl;
      } else if(direction_ == 0) {
        cout << "Particle " << pId << " near horizontal wall, pos: " << d_particlePos[pId * nDim + 1] << endl;
      }
    }
  }*/

  // locally rescale velocities for particles near compressing walls
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  const double *prevEnergy = thrust::raw_pointer_cast(&d_prevEnergy[0]);

  auto adjustLocalParticleVel = [=] __device__(long pId) {
    if(wallLabel[pId] == 1) {
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
    }
    /*double scale = 1.0;
    if(ekin > deltaU) {
      scale = sqrt((ekin - deltaU) / ekin);
    } else {
      scale = sqrt((deltaU - ekin) / ekin);
    }
    #pragma unroll(MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] *= scale;
    }*/
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
  thrust::transform(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, d_squaredVel.begin(), square());
  return mass * thrust::reduce(d_squaredVel.begin(), d_squaredVel.end(), double(0), thrust::plus<double>()) / (firstIndex * nDim);
}

void SP2D::calcParticleEnergyAB() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double *sqVelAB = thrust::raw_pointer_cast(&d_squaredVelAB[0]);
	double *pEnergyAB = thrust::raw_pointer_cast(&d_particleEnergyAB[0]);
	long *flagAB = thrust::raw_pointer_cast(&d_flagAB[0]);
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    kernelCalcParticleEnergyAB<<<dimGrid, dimBlock>>>(pRad, pPos, pVel, sqVelAB, pEnergyAB, flagAB);
    break;
    default:
    break;
  }
}

std::tuple<double, double, long> SP2D::getParticleEnergyAB() {
  calcParticleEnergyAB();
  long numParticlesAB = thrust::reduce(d_flagAB.begin(), d_flagAB.end(), 0, thrust::plus<long>());
  double epot = thrust::reduce(d_particleEnergyAB.begin(), d_particleEnergyAB.end(), double(0), thrust::plus<double>());
  double ekin = 0.5 * thrust::reduce(d_squaredVelAB.begin(), d_squaredVelAB.end(), double(0), thrust::plus<double>());
  return std::make_tuple(epot, ekin, numParticlesAB);
}

void SP2D::calcParticleHeatAB() {
  thrust::fill(d_particleEnergyAB.begin(), d_particleEnergyAB.end(), double(0));
	long *flagAB = thrust::raw_pointer_cast(&d_flagAB[0]);
	const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
	double *pEnergyAB = thrust::raw_pointer_cast(&d_particleEnergyAB[0]);
  if(simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
    // add work done by damping forces
    double s_dt(dt);
    long s_nDim(nDim);
    double s_gamma(this->sim_->gamma);
    auto r = thrust::counting_iterator<long>(0);

    auto addDampingWorkAB = [=] __device__ (long pId) {
      if(flagAB[pId] == 1) {
        #pragma unroll (MAXDIM)
        for (long dim = 0; dim < s_nDim; dim++) {
          pEnergyAB[pId] -= s_dt * s_gamma * pVel[pId * s_nDim + dim] * pVel[pId * s_nDim + dim];
        }
      }
    };

    thrust::for_each(r, r + numParticles, addDampingWorkAB);
    // add work done by white noise
    double s_noise(this->sim_->noise);
    auto t = thrust::counting_iterator<long>(0);
	  const double *rand = thrust::raw_pointer_cast(&(this->sim_->d_rand[0]));

    auto addNoiseWorkAB = [=] __device__ (long pId) {
      if(flagAB[pId] == 1) {
        #pragma unroll (MAXDIM)
        for (long dim = 0; dim < s_nDim; dim++) {
          pEnergyAB[pId] += s_dt * s_noise * rand[pId * s_nDim + dim] * pVel[pId * s_nDim + dim];
        }
      }
    };

    thrust::for_each(t, t + numParticles, addNoiseWorkAB);
    // add work done by self-propulsion if particles are active
    if(simControl.particleType == simControlStruct::particleEnum::active) {
      double s_driving(driving);
      auto s = thrust::counting_iterator<long>(0);
      const double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);

      auto addActiveWorkAB = [=] __device__ (long pId) {
        #pragma unroll (MAXDIM)
        for (long dim = 0; dim < s_nDim; dim++) {
          pEnergyAB[pId] += s_dt * s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId])) * pVel[pId * s_nDim + dim];
        }
      };

      thrust::for_each(r, r + numParticles, addActiveWorkAB);
    }
  }
}

std::tuple<double, double, double, long> SP2D::getParticleWorkAB() {
  calcParticleEnergyAB();
  long numParticlesAB = thrust::reduce(d_flagAB.begin(), d_flagAB.end(), 0, thrust::plus<long>());
  double epot = thrust::reduce(d_particleEnergyAB.begin(), d_particleEnergyAB.end(), double(0), thrust::plus<double>());
  double ekin = 0.5 * thrust::reduce(d_squaredVelAB.begin(), d_squaredVelAB.end(), double(0), thrust::plus<double>());
  calcParticleHeatAB();
  double heat = thrust::reduce(d_particleEnergyAB.begin(), d_particleEnergyAB.end(), double(0), thrust::plus<double>());
  return std::make_tuple(epot, ekin, heat, numParticlesAB);
}

//************************* contacts and neighbors ***************************//
thrust::host_vector<long> SP2D::getParticleNeighbors() {
  thrust::host_vector<long> partNeighborListFromDevice;
  partNeighborListFromDevice = d_partNeighborList;
  return partNeighborListFromDevice;
}

thrust::host_vector<long> SP2D::getVicsekNeighbors() {
  thrust::host_vector<long> vicsekNeighborListFromDevice;
  vicsekNeighborListFromDevice = d_vicsekNeighborList;
  return vicsekNeighborListFromDevice;
}

thrust::host_vector<long> SP2D::getWallNeighbors() {
  thrust::host_vector<long> wallNeighborListFromDevice;
  wallNeighborListFromDevice = d_wallNeighborList;
  return wallNeighborListFromDevice;
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
  switch (simControl.boundaryType) {
    case simControlStruct::boundaryEnum::rough:
    case simControlStruct::boundaryEnum::rigid:
    case simControlStruct::boundaryEnum::mobile:
    case simControlStruct::boundaryEnum::plastic:
    calcWallNeighborList(cutDistance);
    break;
    default:
    break;
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

void SP2D::calcVicsekNeighborList() {
  thrust::fill(d_vicsekMaxNeighborList.begin(), d_vicsekMaxNeighborList.end(), 0);
	thrust::fill(d_vicsekNeighborList.begin(), d_vicsekNeighborList.end(), -1L);
  syncVicsekNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcVicsekNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, Rvicsek);
  // compute maximum number of neighbors per particle
  if(cudaGetLastError()) cout << "SP2D::calcVicsekNeighborList():: cudaGetLastError(): " << cudaGetLastError() << endl;
  vicsekMaxNeighbors = thrust::reduce(d_vicsekMaxNeighborList.begin(), d_vicsekMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncVicsekNeighborsToDevice();
  //cout << "SP2D::calcVicsekNeighborList: vicsekMaxNeighbors: " << vicsekMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( vicsekMaxNeighbors > vicsekNeighborListSize ) {
		vicsekNeighborListSize = pow(2, ceil(std::log2(vicsekMaxNeighbors)));
    //cout << "SP2D::calcVicsekNeighborList: vicsekNeighborListSize: " << vicsekNeighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_vicsekNeighborList.resize(numParticles * vicsekNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_vicsekNeighborList.begin(), d_vicsekNeighborList.end(), -1L);
		syncVicsekNeighborsToDevice();
		kernelCalcVicsekNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, Rvicsek);
	}
}

void SP2D::syncVicsekNeighborsToDevice() {
  cudaDeviceSynchronize();
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_vicsekNeighborListSize, &vicsekNeighborListSize, sizeof(vicsekNeighborListSize));
	cudaMemcpyToSymbol(d_vicsekMaxNeighbors, &vicsekMaxNeighbors, sizeof(vicsekMaxNeighbors));

	long* vicsekMaxNeighborList = thrust::raw_pointer_cast(&d_vicsekMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_vicsekMaxNeighborListPtr, &vicsekMaxNeighborList, sizeof(vicsekMaxNeighborList));

	long* vicsekNeighborList = thrust::raw_pointer_cast(&d_vicsekNeighborList[0]);
	cudaMemcpyToSymbol(d_vicsekNeighborListPtr, &vicsekNeighborList, sizeof(vicsekNeighborList));
  if(cudaGetLastError()) cout << "SP2D::syncVicsekNeighborsToDevice():: cudaGetLastError(): " << cudaGetLastError() << endl;
}

void SP2D::calcWallNeighborList(double cutDistance) {
  thrust::fill(d_wallMaxNeighborList.begin(), d_wallMaxNeighborList.end(), 0);
	thrust::fill(d_wallNeighborList.begin(), d_wallNeighborList.end(), -1L);
  syncWallNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *wPos = thrust::raw_pointer_cast(&d_wallPos[0]);

  kernelCalcParticleWallNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, wPos, cutDistance);
  // compute maximum number of neighbors per particle
  if(cudaGetLastError()) cout << "SP2D::calcWallNeighborList():: cudaGetLastError(): " << cudaGetLastError() << endl;
  wallMaxNeighbors = thrust::reduce(d_wallMaxNeighborList.begin(), d_wallMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncWallNeighborsToDevice();
  //cout << "SP2D::calcWallNeighborList: wallMaxNeighbors: " << wallMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( wallMaxNeighbors > wallNeighborListSize ) {
		wallNeighborListSize = pow(2, ceil(std::log2(wallMaxNeighbors)));
    //cout << "SP2D::calcWallNeighborList: wallNeighborListSize: " << wallNeighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_wallNeighborList.resize(numParticles * wallNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_wallNeighborList.begin(), d_wallNeighborList.end(), -1L);
		syncWallNeighborsToDevice();
		kernelCalcParticleWallNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, wPos, cutDistance);
	}
}

void SP2D::syncWallNeighborsToDevice() {
  cudaDeviceSynchronize();
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_wallNeighborListSize, &wallNeighborListSize, sizeof(wallNeighborListSize));
	cudaMemcpyToSymbol(d_wallMaxNeighbors, &wallMaxNeighbors, sizeof(wallMaxNeighbors));

	long* wallMaxNeighborList = thrust::raw_pointer_cast(&d_wallMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_wallMaxNeighborListPtr, &wallMaxNeighborList, sizeof(wallMaxNeighborList));

	long* wallNeighborList = thrust::raw_pointer_cast(&d_wallNeighborList[0]);
	cudaMemcpyToSymbol(d_wallNeighborListPtr, &wallNeighborList, sizeof(wallNeighborList));
  if(cudaGetLastError()) cout << "SP2D::syncWallNeighborsToDevice():: cudaGetLastError(): " << cudaGetLastError() << endl;
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

//***************************** Langevin integrators ******************************//
void SP2D::initSoftParticleLangevin(double Temp, double gamma, bool readState) {
  switch (simControl.noiseType) {
    case simControlStruct::noiseEnum::langevin1:
    this->sim_ = new SoftParticleLangevin(this, SimConfig(Temp, 0, 0));
    cout << "SP2D::initSoftParticleLangevin:: noiseType 1";
    break;
    case simControlStruct::noiseEnum::langevin2:
    this->sim_ = new SoftParticleLangevin2(this, SimConfig(Temp, 0, 0));
    this->sim_->d_rando.resize(numParticles * nDim);
    thrust::fill(this->sim_->d_rando.begin(), this->sim_->d_rando.end(), double(0));
    cout << "SP2D::initSoftParticleLangevin:: noiseType 2";
    break;
    case simControlStruct::noiseEnum::brownian:
    this->sim_ = new SoftParticleBrownian(this, SimConfig(Temp, 0, 0));
    cout << "SP2D::initSoftParticleLangevin:: Brownian integrator";
    break;
    case simControlStruct::noiseEnum::drivenBrownian:
    this->sim_ = new SoftParticleDrivenBrownian(this, SimConfig(Temp, 0, 0));
    cout << "SP2D::initSoftParticleLangevin:: Driven Brownian integrator";
    break;
  }
  this->sim_->gamma = gamma;
  this->sim_->noise = sqrt(2. * Temp * gamma / dt);
  this->sim_->d_rand.resize(numParticles * nDim);
  thrust::fill(this->sim_->d_rand.begin(), this->sim_->d_rand.end(), double(0));
  resetLastPositions();
  setInitialPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << " current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void SP2D::softParticleLangevinLoop(bool conserve) {
  this->sim_->integrate();
  if(conserve == true) {
    this->sim_->conserveMomentum();
  }
}

void SP2D::initSoftParticleLangevinSubset(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel) {
  this->sim_ = new SoftParticleLangevinSubset(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noise = sqrt(2. * Temp * gamma / dt);
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  thrust::fill(this->sim_->d_rand.begin(), this->sim_->d_rand.end(), double(0));
  thrust::fill(this->sim_->d_rando.begin(), this->sim_->d_rando.end(), double(0));
  // subset variables
  this->sim_->firstIndex = firstIndex;
  this->sim_->mass = mass;
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

void SP2D::softParticleLangevinSubsetLoop() {
  this->sim_->integrate();
}

void SP2D::initSoftParticleLangevinExtField(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevinExtField(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noise = sqrt(2. * Temp * gamma / dt);
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  thrust::fill(this->sim_->d_rand.begin(), this->sim_->d_rand.end(), double(0));
  thrust::fill(this->sim_->d_rando.begin(), this->sim_->d_rando.end(), double(0));
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
  this->sim_->noise = sqrt(2. * Temp * gamma / dt);
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  thrust::fill(this->sim_->d_rand.begin(), this->sim_->d_rand.end(), double(0));
  thrust::fill(this->sim_->d_rando.begin(), this->sim_->d_rando.end(), double(0));
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
  this->sim_->noise = sqrt(2. * Temp * gamma / dt);
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  thrust::fill(this->sim_->d_rand.begin(), this->sim_->d_rand.end(), double(0));
  thrust::fill(this->sim_->d_rando.begin(), this->sim_->d_rando.end(), double(0));
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

void SP2D::rescaleParticleVelocity(double Temp) {
  double scale = sqrt(Temp / getParticleTemperature());
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

  auto rescaleParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] *= scale;
    }
  };

  thrust::for_each(r, r + numParticles, rescaleParticleVel);
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

//************************* Nose-Hoover integrator ***************************//
void SP2D::getNoseHooverParams(double &mass, double &damping) {
  mass = this->sim_->mass;
  damping = this->sim_->gamma;
  //cout << "SP2D::getNoseHooverParams:: damping: " << this->sim_->gamma << endl;
}

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
  damping1 = this->sim_->gamma1;
  damping2 = this->sim_->gamma2;
  //cout << "SP2D::getNoseHooverParams:: damping: " << this->sim_->gamma << endl;
}

//********************** double T Nose-Hoover integrator *********************//
void SP2D::initSoftParticleDoubleNoseHoover(double Temp1, double Temp2, double mass, double gamma1, double gamma2, bool readState) {
  this->sim_ = new SoftParticleDoubleNoseHoover(this, SimConfig(Temp1, 0, Temp2));
  this->sim_->mass = mass;
  this->sim_->gamma1 = gamma1;
  this->sim_->gamma2 = gamma2;
  resetLastPositions();
  setInitialPositions();
  shift = true;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  std::tuple<double, double, double> Temps = getParticleT1T2();
  cout << "SP2D::initSoftParticleDoubleNoseHoover:: T1: " << setprecision(12) << get<0>(Temps) << " T2: " << get<1>(Temps) << " T: " << get<2>(Temps) << endl;
  cout << " mass: " << this->sim_->mass << ", damping1: " << this->sim_->gamma1 << " damping2: " << this->sim_->gamma2 << endl;
}

void SP2D::softParticleDoubleNoseHooverLoop() {
  this->sim_->integrate();
}
