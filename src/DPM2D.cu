//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// FUNCTION DECLARATIONS

#include "../include/DPM2D.h"
#include "../include/DPM2DKernel.cuh"
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

//************************** dpm object definition ***************************//
DPM2D::DPM2D(long nParticles, long dim, long nVertexPerParticle) {
  // default values
  srand48(time(0));
  dimBlock = 256;
  nDim = dim;
  numParticles = nParticles;
  numVertexPerParticle = nVertexPerParticle;
  // the default is monodisperse size distribution
  // same number of vertices per particle
  numVertices = numParticles * numVertexPerParticle;
  setDimBlock(dimBlock);
  setNDim(nDim);
  setNumParticles(numParticles);
  setNumVertexPerParticle(numVertexPerParticle);
  setNumVertices(numVertices);
  // set force paramters to zero
  // TODO: maybe initialize these?
  dt = 0.;
  rho0 = 0.;
  ea = 0.;
	el = 0.;
	eb = 0.;
	ec = 0.;
	l1 = 0.;
	l2 = 0.;
  d_boxSize.resize(nDim);
  thrust::fill(d_boxSize.begin(), d_boxSize.end(), double(1));
  d_stress.resize(nDim * nDim);
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  d_numVertexInParticleList.resize(numParticles);
  setMonoSizeDistribution();
  d_firstVertexInParticleId.resize(numParticles);
  initParticleIdList();
  // particle variables
  initParticleVariables(numParticles);
  // particle dynmaical variables
  initParticleDynamicalVariables(numParticles);
  // vertex shape variables
  initVertexVariables(numVertices);
  // vertex dynamical variables
  initDynamicalVariables(numVertices);
  // initialize contacts and neighbors
  initContacts(numParticles);
  initNeighbors(numVertices);
  syncNeighborsToDevice();
  initParticleNeighbors(numParticles);
  syncParticleNeighborsToDevice();
}

DPM2D::~DPM2D() {
	// clear all vectors and pointers
	d_boxSize.clear();
  d_stress.clear();
  d_numVertexInParticleList.clear();
  d_firstVertexInParticleId.clear();
  d_particleIdList.clear();
  d_a0.clear();
  d_rad.clear();
  d_l0.clear();
  d_theta0.clear();
  d_length.clear();
  d_l0Vel.clear();
  d_area.clear();
  d_perimeter.clear();
  d_particleRad.clear();
  d_particlePos.clear();
  d_particleVel.clear();
  d_particleForce.clear();
  d_particleEnergy.clear();
  d_particleTorque.clear();
  d_particleAngvel.clear();
  d_particleAngle.clear();
  d_particleInitAngle.clear();
  // dynamical variables
  d_pos.clear();
  d_vel.clear();
  d_force.clear();
  d_energy.clear();
  d_torque.clear();
  d_initialPos.clear();
  d_particleInitPos.clear();
  d_particleDelta.clear();
  d_particleDeltaAngle.clear();
  d_particlePreviousPos.clear();
  // contacts and neighbors
  d_contactList.clear();
  d_numContacts.clear();
  d_contactVectorList.clear();
  d_neighborList.clear();
  d_maxNeighborList.clear();
  d_numPartNeighbors.clear();
  d_partNeighborList.clear();
  d_partMaxNeighborList.clear();
}

void DPM2D::initParticleVariables(long numParticles_) {
  d_a0.resize(numParticles_);
  d_area.resize(numParticles_);
  d_perimeter.resize(numParticles_);
  d_particleAngle.resize(numParticles_);
  d_perParticleStress.resize(numParticles_ * nDim * nDim);
  thrust::fill(d_a0.begin(), d_a0.end(), double(0));
  thrust::fill(d_area.begin(), d_area.end(), double(0));
  thrust::fill(d_perimeter.begin(), d_perimeter.end(), double(0));
  thrust::fill(d_particleAngle.begin(), d_particleAngle.end(), double(0));
  thrust::fill(d_perParticleStress.begin(), d_perParticleStress.end(), double(0));
}

void DPM2D::initParticleDynamicalVariables(long numParticles_) {
  d_particleRad.resize(numParticles_);
  d_particlePos.resize(numParticles_ * nDim);
  d_particleVel.resize(numParticles_ * nDim);
  d_particleForce.resize(numParticles_ * nDim);
  d_particleEnergy.resize(numParticles_);
  thrust::fill(d_particleRad.begin(), d_particleRad.end(), double(0));
  thrust::fill(d_particlePos.begin(), d_particlePos.end(), double(0));
  thrust::fill(d_particleVel.begin(), d_particleVel.end(), double(0));
  thrust::fill(d_particleForce.begin(), d_particleForce.end(), double(0));
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
}

void DPM2D::initRotationalVariables(long numVertices_, long numParticles_) {
  d_torque.resize(numVertices_);
  d_particleAngvel.resize(numParticles_);
  d_particleTorque.resize(numParticles_);
  thrust::fill(d_torque.begin(), d_torque.end(), double(0));
  thrust::fill(d_particleAngvel.begin(), d_particleAngvel.end(), double(0));
  thrust::fill(d_particleTorque.begin(), d_particleTorque.end(), double(0));
}

void DPM2D::initVertexVariables(long numVertices_) {
  d_rad.resize(numVertices_);
  d_l0.resize(numVertices_);
  d_theta0.resize(numVertices_);
  d_length.resize(numVertices_);
  thrust::fill(d_rad.begin(), d_rad.end(), double(0));
  thrust::fill(d_l0.begin(), d_l0.end(), double(0));
  thrust::fill(d_theta0.begin(), d_theta0.end(), double(0));
  thrust::fill(d_length.begin(), d_length.end(), double(0));
}

void DPM2D::initDynamicalVariables(long numVertices_) {
  d_pos.resize(numVertices_ * nDim);
  d_vel.resize(numVertices_ * nDim);
  d_force.resize(numVertices_ * nDim);
  d_energy.resize(numVertices_);
  d_lastPos.resize(numVertices_ * nDim);
  d_disp.resize(numVertices_);
  thrust::fill(d_pos.begin(), d_pos.end(), double(0));
  thrust::fill(d_vel.begin(), d_vel.end(), double(0));
  thrust::fill(d_force.begin(), d_force.end(), double(0));
  thrust::fill(d_energy.begin(), d_energy.end(), double(0));
  thrust::fill(d_lastPos.begin(), d_lastPos.end(), double(0));
  thrust::fill(d_disp.begin(), d_disp.end(), double(0));
}

void DPM2D::initDeltaVariables(long numVertices_, long numParticles_) {
  d_initialPos.resize(numVertices_ * nDim);
  d_delta.resize(numVertices_ * nDim);
  d_particleInitPos.resize(numParticles_ * nDim);
  d_particleDelta.resize(numParticles_ * nDim);
  d_particleDisp.resize(numParticles_);
  d_particleInitAngle.resize(numParticles_);
  d_particleDeltaAngle.resize(numParticles_);
  thrust::fill(d_initialPos.begin(), d_initialPos.end(), double(0));
  thrust::fill(d_delta.begin(), d_delta.end(), double(0));
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
  thrust::fill(d_particleDisp.begin(), d_particleDisp.end(), double(0));
  thrust::fill(d_particleInitAngle.begin(), d_particleInitAngle.end(), double(0));
  thrust::fill(d_particleDeltaAngle.begin(), d_particleDeltaAngle.end(), double(0));
}

void DPM2D::initContacts(long numParticles_) {
  long maxContacts = 8 * nDim; // guess
  d_numContacts.resize(numParticles_);
  d_contactList.resize(numParticles_ * maxContacts);
  d_numPartNeighbors.resize(numParticles_);
  d_partNeighborList.resize(numParticles_ * maxContacts);
  d_contactVectorList.resize(numParticles_ * nDim * maxContacts);
  thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
  thrust::fill(d_contactList.begin(), d_contactList.end(), double(0));
  thrust::fill(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L);
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), double(0));
  thrust::fill(d_contactVectorList.begin(), d_contactVectorList.end(), double(0));
}

void DPM2D::initNeighbors(long numVertices_) {
  neighborListSize = 0;
  maxNeighbors = 0;
  d_neighborList.resize(numVertices_);
  d_maxNeighborList.resize(numVertices_);
  thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
  thrust::fill(d_maxNeighborList.begin(), d_maxNeighborList.end(), maxNeighbors);
}

void DPM2D::initParticleNeighbors(long numParticles_) {
  partNeighborListSize = 0;
  partMaxNeighbors = 0;
  d_partNeighborList.resize(numParticles_);
  d_partMaxNeighborList.resize(numParticles_);
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), partMaxNeighbors);
}


void DPM2D::initParticleIdList() {
  long countVertices = 0;
  d_particleIdList.resize(numVertices);
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_firstVertexInParticleId[particleId] = countVertices;
    for(long vertexInPartId = 0; vertexInPartId < d_numVertexInParticleList[particleId]; vertexInPartId++) {
      d_particleIdList[countVertices] = particleId;
			countVertices += 1;
		}
  }
  long* firstVertexInParticleId = thrust::raw_pointer_cast(&d_firstVertexInParticleId[0]);
  cudaMemcpyToSymbol(d_firstVertexInParticleIdPtr, &firstVertexInParticleId, sizeof(firstVertexInParticleId));

  long* particleIdList = thrust::raw_pointer_cast(&d_particleIdList[0]);
  cudaMemcpyToSymbol(d_particleIdListPtr, &particleIdList, sizeof(particleIdList));
}

//**************************** setters and getters ***************************//
// TODO: add error checks for all the getters and setters
void DPM2D::setDimBlock(long dimBlock_) {
	dimBlock = dimBlock_;
	dimGrid = (numVertices + dimBlock - 1) / dimBlock;
	partDimGrid = (numParticles + dimBlock - 1) / dimBlock;
  cudaError err = cudaMemcpyToSymbol(d_dimBlock, &dimBlock, sizeof(dimBlock));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
  err = cudaMemcpyToSymbol(d_dimGrid, &dimGrid, sizeof(dimGrid));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
  err = cudaMemcpyToSymbol(d_partDimGrid, &partDimGrid, sizeof(partDimGrid));
  if(err != cudaSuccess) {
    cout << "cudaMemcpyToSymbol Error: "<< cudaGetErrorString(err) << endl;
  }
}

long DPM2D::getDimBlock() {
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

void DPM2D::setNDim(long nDim_) {
  nDim = nDim_;
  cudaMemcpyToSymbol(d_nDim, &nDim, sizeof(nDim));
}

long DPM2D::getNDim() {
  long nDimFromDevice;
  cudaMemcpyFromSymbol(&nDimFromDevice, d_nDim, sizeof(d_nDim));
	return nDimFromDevice;
}

void DPM2D::setNumParticles(long numParticles_) {
  numParticles = numParticles_;
  cudaMemcpyToSymbol(d_numParticles, &numParticles, sizeof(numParticles));
}

long DPM2D::getNumParticles() {
  long numParticlesFromDevice;
  cudaMemcpyFromSymbol(&numParticlesFromDevice, d_numParticles, sizeof(d_numParticles));
	return numParticlesFromDevice;
}

void DPM2D::setNumVertices(long numVertices_) {
  numVertices = numVertices_;
  cudaMemcpyToSymbol(d_numVertices, &(numVertices), sizeof(numVertices));
  setDimBlock(dimBlock); // recalculate dimGrid
}

long DPM2D::getNumVertices() {
  long numVerticesFromDevice;
  cudaMemcpyFromSymbol(&numVerticesFromDevice, d_numVertices, sizeof(d_numVertices));
	return numVerticesFromDevice;
}

void DPM2D::setNumVertexPerParticle(long numVertexPerParticle_) {
  numVertexPerParticle = numVertexPerParticle_;
  cudaMemcpyToSymbol(d_numVertexPerParticle, &numVertexPerParticle, sizeof(numVertexPerParticle));
}

long DPM2D::getNumVertexPerParticle() {
  long numVertexPerParticleFromDevice;
  cudaMemcpyFromSymbol(&numVertexPerParticleFromDevice, d_numVertexPerParticle, sizeof(d_numVertexPerParticle));
  return numVertexPerParticleFromDevice;
}

void DPM2D::setNumVertexInParticleList(thrust::host_vector<long> &numVertexInParticleList_) {
  if(numVertexInParticleList_.size() == ulong(numParticles)) {
    d_numVertexInParticleList = numVertexInParticleList_;
    long* numVertexInParticleList = thrust::raw_pointer_cast(&(d_numVertexInParticleList[0]));
    cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));
  } else {
    cout << "DPM2D::setNumVertexInParticleList: size of numVertexInParticleList does not match numParticles" << endl;
  }
}

thrust::host_vector<long> DPM2D::getNumVertexInParticleList() {
  thrust::host_vector<long> numVertexInParticleListFromDevice;
  if(d_numVertexInParticleList.size() == ulong(numParticles)) {
    cudaMemcpyFromSymbol(&d_numVertexInParticleList, d_numVertexInParticleListPtr, sizeof(d_numVertexInParticleListPtr));
    numVertexInParticleListFromDevice = d_numVertexInParticleList;
  } else {
    cout << "DPM2D::getNumVertexInParticleList: size of numVertexInParticleList from device does not match numParticles" << endl;
  }
  return numVertexInParticleListFromDevice;
}

// the length scale is always set to be the sqrt of the first particle area
void DPM2D::setLengthScale() {
  rho0 = sqrt((thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()))/numParticles); // set dimensional factor
  //cout << " lengthscale: " << rho0 << endl;
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

void DPM2D::setParticleLengthScale() {
  rho0 = thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>())/numParticles; // set dimensional factor
  cout << " lengthscale: " << rho0 << endl;
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

void DPM2D::setLengthScaleToOne() {
  rho0 = 1.; // for soft particles
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

//TODO: error messages for all the vector getters and setters
void DPM2D::setBoxSize(thrust::host_vector<double> &boxSize_) {
  if(boxSize_.size() == ulong(nDim)) {
    d_boxSize = boxSize_;
    double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
    cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  } else {
    cout << "DPM2D::setBoxSize: size of boxSize does not match nDim" << endl;
  }
}

thrust::host_vector<double> DPM2D::getBoxSize() {
  thrust::host_vector<double> boxSizeFromDevice;
  if(d_boxSize.size() == ulong(nDim)) {
    cudaMemcpyFromSymbol(&d_boxSize, d_boxSizePtr, sizeof(d_boxSizePtr));
    boxSizeFromDevice = d_boxSize;
  } else {
    cout << "DPM2D::getBoxSize: size of boxSize from device does not match nDim" << endl;
  }
  return boxSizeFromDevice;
}

//**************************** shape variables *******************************//
void DPM2D::setVertexRadii(thrust::host_vector<double> &rad_) {
  d_rad = rad_;
}

thrust::host_vector<double> DPM2D::getVertexRadii() {
  thrust::host_vector<double> radFromDevice;
  radFromDevice = d_rad;
  return radFromDevice;
}

double DPM2D::getMaxRadius() {
  return double(thrust::reduce(d_rad.begin(), d_rad.end(), double(-1), thrust::maximum<double>()));
}

void DPM2D::setRestAreas(thrust::host_vector<double> &a0_) {
  d_a0 = a0_;
}

thrust::host_vector<double> DPM2D::getRestAreas() {
  thrust::host_vector<double> a0FromDevice;
  a0FromDevice = d_a0;
  return a0FromDevice;
}

void DPM2D::setRestLengths(thrust::host_vector<double> &l0_) {
  d_l0 = l0_;
}

thrust::host_vector<double> DPM2D::getRestLengths() {
  thrust::host_vector<double> l0FromDevice;
  l0FromDevice = d_l0;
  return l0FromDevice;
}

void DPM2D::setRestAngles(thrust::host_vector<double> &theta0_) {
  d_theta0 = theta0_;
}

thrust::host_vector<double> DPM2D::getRestAngles() {
  thrust::host_vector<double> theta0FromDevice;
  theta0FromDevice = d_theta0;
  return theta0FromDevice;
}

thrust::host_vector<double> DPM2D::getSegmentLengths() {
  thrust::host_vector<double> lengthFromDevice;
  lengthFromDevice = d_length;
  return lengthFromDevice;
}

void DPM2D::setAreas(thrust::host_vector<double> &area_) {
  d_area = area_;
}

thrust::host_vector<double> DPM2D::getAreas() {
  thrust::host_vector<double> areaFromDevice;
  areaFromDevice = d_area;
  return areaFromDevice;
}

thrust::host_vector<double> DPM2D::getPerimeters() {
  thrust::host_vector<double> perimeterFromDevice;
  perimeterFromDevice = d_perimeter;
  return perimeterFromDevice;
}

void DPM2D::calcParticlesShape() {
  // area and perimeter pointers
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *length = thrust::raw_pointer_cast(&d_length[0]);
  double *area = thrust::raw_pointer_cast(&d_area[0]);
  double *perimeter = thrust::raw_pointer_cast(&d_perimeter[0]);

  kernelCalcParticlesShape<<<dimGrid, dimBlock>>>(pos, length, area, perimeter);
}

void DPM2D::calcParticlesPositions() {
  // vertex and particle position pointers
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);

  kernelCalcParticlesPositions<<<dimGrid, dimBlock>>>(pos, particlePos);
}

void DPM2D::setDefaultParticleRadii() {
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] = sqrt(d_a0[particleId]/PI);
  }
}

void DPM2D::setParticleRadii(thrust::host_vector<double> &particleRad_) {
  d_particleRad = particleRad_;
}

thrust::host_vector<double> DPM2D::getParticleRadii() {
  thrust::host_vector<double> particleRadFromDevice;
  particleRadFromDevice = d_particleRad;
  return particleRadFromDevice;
}

void DPM2D::setParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
}

void DPM2D::setPBCParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
  // check pbc
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  kernelCheckParticlePBC<<<partDimGrid, dimBlock>>>(pPosPBC, pPos);
  // copy to device
  d_particlePos = d_particlePosPBC;
}

thrust::host_vector<double> DPM2D::getParticlePositions() {
  thrust::host_vector<double> particlePosFromDevice;
  particlePosFromDevice = d_particlePos;
  return particlePosFromDevice;
}

thrust::host_vector<double> DPM2D::getPBCParticlePositions() {
  // check pbc
  thrust::device_vector<double> d_particlePosPBC(d_particlePos.size());
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pPosPBC = thrust::raw_pointer_cast(&d_particlePosPBC[0]);
  kernelCheckParticlePBC<<<partDimGrid, dimBlock>>>(pPosPBC, pPos);
  // copy to host
  thrust::host_vector<double> particlePosFromDevice;
  particlePosFromDevice = d_particlePosPBC;
  return particlePosFromDevice;
}

void DPM2D::resetPreviousPositions() {
  d_particlePreviousPos = getParticlePositions();
}

void DPM2D::resetLastPositions() {
  d_lastPos = getVertexPositions();
}

thrust::host_vector<double> DPM2D::getPreviousPositions() {
  thrust::host_vector<double> previousPosFromDevice;
  previousPosFromDevice = d_particlePreviousPos;
  return previousPosFromDevice;
}

void DPM2D::setParticleVelocities(thrust::host_vector<double> &particleVel_) {
  d_particleVel = particleVel_;
}

thrust::host_vector<double> DPM2D::getParticleVelocities() {
  thrust::host_vector<double> particleVelFromDevice;
  particleVelFromDevice = d_particleVel;
  return particleVelFromDevice;
}

void DPM2D::setParticleForces(thrust::host_vector<double> &particleForce_) {
  d_particleForce = particleForce_;
}

thrust::host_vector<double> DPM2D::getParticleForces() {
  thrust::host_vector<double> particleForceFromDevice;
  particleForceFromDevice = d_particleForce;
  return particleForceFromDevice;
}

thrust::host_vector<double> DPM2D::getParticleEnergies() {
  thrust::host_vector<double> particleEnergyFromDevice;
  particleEnergyFromDevice = d_particleEnergy;
  return particleEnergyFromDevice;
}

double DPM2D::getMeanParticleSize() {
  return sqrt(thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()) / (PI * numParticles));
}

double DPM2D::getMeanParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>()) / numParticles;
}

double DPM2D::getMinParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(1), thrust::minimum<double>());
}

void DPM2D::setParticleAngles(thrust::host_vector<double> &particleAngle_) {
  d_particleAngle = particleAngle_;
}

thrust::host_vector<double> DPM2D::getParticleAngles() {
  thrust::host_vector<double> particleAngleFromDevice;
  particleAngleFromDevice = d_particleAngle;
  return particleAngleFromDevice;
}

//************************** dynamical variables *****************************//
void DPM2D::setVertexPositions(thrust::host_vector<double> &pos_) {
  d_pos = pos_;
}

thrust::host_vector<double> DPM2D::getVertexPositions() {
  thrust::host_vector<double> posFromDevice;
  posFromDevice = d_pos;
  return posFromDevice;
}

void DPM2D::setVertexVelocities(thrust::host_vector<double> &vel_) {
  d_vel = vel_;
}

thrust::host_vector<double> DPM2D::getVertexVelocities() {
  thrust::host_vector<double> velFromDevice;
  velFromDevice = d_vel;
  return velFromDevice;
}

void DPM2D::setVertexForces(thrust::host_vector<double> &force_) {
  d_force = force_;
}

thrust::host_vector<double> DPM2D::getVertexForces() {
  thrust::host_vector<double> forceFromDevice;
  forceFromDevice = d_force;
  return forceFromDevice;
}

void DPM2D::setVertexTorques(thrust::host_vector<double> &torque_) {
  d_torque = torque_;
}

thrust::host_vector<double> DPM2D::getVertexTorques() {
  thrust::host_vector<double> torqueFromDevice;
  torqueFromDevice = d_torque;
  return torqueFromDevice;
}

thrust::host_vector<double> DPM2D::getStressTensor() {
  calcStressTensor();
  thrust::host_vector<double> stressFromDevice;
  stressFromDevice = d_stress;
  return stressFromDevice;
}

thrust::host_vector<double> DPM2D::getPerParticleStressTensor() {
  calcPerParticleStressTensor();
  thrust::host_vector<double> perParticleStressFromDevice;
  perParticleStressFromDevice = d_perParticleStress;
  return perParticleStressFromDevice;
}

double DPM2D::getPressure() {
  calcStressTensor();
  double pressure = 0;
  for (long dim = 0; dim < nDim; dim++) {
    pressure += d_stress[dim * nDim + dim];
  }
  return pressure / (nDim * numVertices);
}

// return the sum of force magnitudes
double DPM2D::getTotalForceMagnitude() {
  thrust::device_vector<double> forceSquared(d_force.size());
  // compute squared velocities
  thrust::transform(d_force.begin(), d_force.end(), forceSquared.begin(), square());
  // sum squares
  double totalForceMagnitude = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(0), thrust::plus<double>()) / (numVertices * nDim));
  forceSquared.clear();
  return totalForceMagnitude;
}

// return the maximum force magnitude
double DPM2D::getMaxUnbalancedForce() {
  thrust::device_vector<double> forceSquared(d_force.size());
  // compute squared velocities
  thrust::transform(d_force.begin(), d_force.end(), forceSquared.begin(), square());

  double maxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
  return maxUnbalancedForce;
}

thrust::host_vector<long> DPM2D::getMaxNeighborList() {
  thrust::host_vector<long> maxNeighborListFromDevice;
  maxNeighborListFromDevice = d_maxNeighborList;
  return maxNeighborListFromDevice;
}

thrust::host_vector<long> DPM2D::getNeighbors() {
  thrust::host_vector<long> neighborListFromDevice;
  neighborListFromDevice = d_neighborList;
  return neighborListFromDevice;
}

thrust::host_vector<long> DPM2D::getContacts() {
  thrust::host_vector<long> contactListFromDevice;
  contactListFromDevice = d_contactList;
  return contactListFromDevice;
}

void DPM2D::printNeighbors() {
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    cout << "vertexId: " << vertexId << " list of neighbors: ";
    for (long neighborId = 0; neighborId < d_maxNeighborList[vertexId]; neighborId++) {
      cout << d_neighborList[vertexId * neighborListSize + neighborId] << " ";
    }
    cout << endl;
  }
}

void DPM2D::printContacts() {
  for (long particleId = 0; particleId < numParticles; particleId++) {
    cout << "particleId: " << particleId << " list of contacts: ";
    for (long contactId = 0; contactId < d_numContacts[particleId]; contactId++) {
      cout << d_contactList[particleId * contactLimit + contactId] << " ";
    }
    cout << endl;
  }
}

double DPM2D::getPotentialEnergy() {
  return thrust::reduce(d_energy.begin(), d_energy.end(), double(0), thrust::plus<double>());
}

double DPM2D::getSmoothPotentialEnergy() {
  // the interaction energy is saved on the particle level and the shape energy
  // is saved on the vertex level for smooth interaction between vertices of different particles
  double totalEnergy = 0;
  totalEnergy = thrust::reduce(d_energy.begin(), d_energy.end(), double(0), thrust::plus<double>());
  totalEnergy += thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
  return totalEnergy;
}

double DPM2D::getKineticEnergy() {
  thrust::device_vector<double> velSquared(d_vel.size());
  // compute squared velocities
  thrust::transform(d_vel.begin(), d_vel.end(), velSquared.begin(), square());
  // sum squares
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end());
}

double DPM2D::getTemperature() {
  return 2. * getKineticEnergy() / (nDim * numVertices);
}

double DPM2D::getTotalEnergy() {
  double etot = getPotentialEnergy();
  etot += getKineticEnergy();
  return etot;
}

double DPM2D::getPhi() {
  double phi = double(thrust::reduce(d_area.begin(), d_area.end(), double(0), thrust::plus<double>()));
  // add effective vertex areas
  thrust::device_vector<double> d_vertexArea(d_area.size());
  double *vertexArea = thrust::raw_pointer_cast(&d_vertexArea[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  kernelCalcVertexArea<<<dimGrid,dimBlock>>>(rad, vertexArea);
  phi += PI * thrust::reduce(d_vertexArea.begin(), d_vertexArea.end(), double(0), thrust::plus<double>());
  return phi / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getPreferredPhi() {
  double phi = double(thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()));
  // add effective vertex areas
  thrust::device_vector<double> d_vertexArea(d_area.size());
  double *vertexArea = thrust::raw_pointer_cast(&d_vertexArea[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  kernelCalcVertexArea<<<dimGrid,dimBlock>>>(rad, vertexArea);
  phi += PI * thrust::reduce(d_vertexArea.begin(), d_vertexArea.end(), double(0), thrust::plus<double>());
  return phi / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getParticlePhi() {
  thrust::device_vector<double> d_radSquared(numParticles);
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radSquared.begin(), square());
  return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::get3DParticlePhi() {
  thrust::device_vector<double> d_volume(numParticles);
  thrust::fill(d_volume.begin(), d_volume.end(), double(1));
  long p_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *volume = thrust::raw_pointer_cast(&d_volume[0]);
  const double *rad = thrust::raw_pointer_cast(&d_particleRad[0]);

  auto computeVolume = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < p_nDim; dim++) {
      volume[particleId] *= rad[particleId];
    }
  };

  thrust::for_each(r, r + numParticles, computeVolume);
  return thrust::reduce(d_volume.begin(), d_volume.end(), double(0), thrust::plus<double>()) * 3 * PI / (4 * d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
}

double DPM2D::getVertexMSD() {
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *initialPos = thrust::raw_pointer_cast(&d_initialPos[0]);
  double *delta = thrust::raw_pointer_cast(&d_delta[0]);
  kernelCalcVertexDistanceSq<<<dimGrid,dimBlock>>>(pos, initialPos, delta);
  return thrust::reduce(d_delta.begin(), d_delta.end(), double(0), thrust::plus<double>()) / (numVertices * d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getMaxDisplacement() {
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *lastPos = thrust::raw_pointer_cast(&d_lastPos[0]);
  double *disp = thrust::raw_pointer_cast(&d_disp[0]);
  kernelCalcVertexDisplacement<<<dimGrid,dimBlock>>>(pos, lastPos, disp);
  return thrust::reduce(d_disp.begin(), d_disp.end(), double(-1), thrust::maximum<double>());
}

double DPM2D::getParticleMSD() {
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelCalcParticleDistanceSq<<<partDimGrid,dimBlock>>>(particlePos, particleInitPos, particleDelta);
  return thrust::reduce(d_particleDelta.begin(), d_particleDelta.end(), double(0), thrust::plus<double>()) / (numParticles * d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getParticleMaxDisplacement() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pPrevPos = thrust::raw_pointer_cast(&d_particlePreviousPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<partDimGrid,dimBlock>>>(pPos, pPrevPos, pDisp);
  return thrust::reduce(d_particleDisp.begin(), d_particleDisp.end(), double(-1), thrust::maximum<double>());
  //auto r = thrust::counting_iterator<long>(0);
  //const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  //const double *pPrevPos = thrust::raw_pointer_cast(&d_particlePreviousPos[0]);

	//auto perParticleDistance = [=] __device__ (int i) {
	//	return calcDistance(&pPos[i*d_nDim], &pPrevPos[i*d_nDim]);
	//};

	//return double( thrust::transform_reduce( r, r + numParticles, perParticleDistance, double(-1), thrust::maximum<double>()) );
}

double DPM2D::getDeformableWaveNumber() {
  return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getPhi() / (PI * numParticles)));
}

double DPM2D::getSoftWaveNumber() {
  if(nDim == 2) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getParticlePhi() / (PI * numParticles)));
  } else if(nDim == 3) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * get3DParticlePhi() / (PI * numParticles)));
  } else {
    cout << "DPM2D::getSoftWaveNumber: this function works only for dim = 2 and 3" << endl;
    return 0;
  }
}

double DPM2D::getVertexISF() {
  double vertexWaveNumber = PI / (2 * d_rad[0]);
  thrust::device_vector<double> d_vertexSF(numVertices);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *initialPos = thrust::raw_pointer_cast(&d_initialPos[0]);
  double *vertexSF = thrust::raw_pointer_cast(&d_vertexSF[0]);
  kernelCalcVertexScatteringFunction<<<dimGrid,dimBlock>>>(pos, initialPos, vertexSF, vertexWaveNumber);
  return thrust::reduce(d_vertexSF.begin(), d_vertexSF.end(), double(0), thrust::plus<double>()) / numVertices;
}

double DPM2D::getParticleISF(double waveNumber_) {
  thrust::device_vector<double> d_particleSF(numParticles);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleSF = thrust::raw_pointer_cast(&d_particleSF[0]);
  kernelCalcParticleScatteringFunction<<<partDimGrid,dimBlock>>>(particlePos, particleInitPos, particleSF, waveNumber_);
  return thrust::reduce(d_particleSF.begin(), d_particleSF.end(), double(0), thrust::plus<double>()) / numParticles;
}

double DPM2D::getHexaticOrderParameter() {
  thrust::device_vector<double> d_psi6(numParticles);
  thrust::fill(d_psi6.begin(), d_psi6.end(), double(0));
  double *psi6 = thrust::raw_pointer_cast(&d_psi6[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  kernelCalcHexaticOrderParameter<<<dimGrid,dimBlock>>>(particlePos, psi6);
  return thrust::reduce(d_psi6.begin(), d_psi6.end(), double(0), thrust::plus<double>()) / numParticles;
}

double DPM2D::getAreaFluctuation() {
  thrust::device_vector<double> deltaA(d_area.size());
  thrust::device_vector<double> deltaASq(d_area.size());
  thrust::fill(deltaA.begin(), deltaA.end(), double(0));
  thrust::transform(d_area.begin(), d_area.end(), d_a0.begin(), deltaA.begin(), thrust::minus<double>());
  thrust::transform(deltaA.begin(), deltaA.end(), deltaASq.begin(), square());
  return sqrt(thrust::reduce(deltaASq.begin(), deltaASq.end(), double(0), thrust::plus<double>()) / numParticles);
}

//************************ initilization functions ***************************//
void DPM2D::setMonoSizeDistribution() {
  thrust::fill(d_numVertexInParticleList.begin(), d_numVertexInParticleList.end(), numVertexPerParticle);
  long* numVertexInParticleList = thrust::raw_pointer_cast(&d_numVertexInParticleList[0]);
  cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));
}

//void DPM2D::setBiSizeDistribution();

void DPM2D::setPolyRandomSoftParticles(double phi0, double polyDispersity) {
  double r1, r2, randNum, mean, sigma, scale, boxLength = 1.;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = exp(mean + randNum * sigma);
    d_a0[particleId] = PI * d_particleRad[particleId] * d_particleRad[particleId];
  }
  scale = sqrt(getParticlePhi() / phi0); // sqrt for 2d
  for (long dim = 0; dim < nDim; dim++) {
    d_boxSize[dim] = boxLength;
  }
  double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
  cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  // extract random positions
  double areaSum = 0;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] /= scale;
    d_a0[particleId] = PI * d_particleRad[particleId] * d_particleRad[particleId];
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
    areaSum += d_a0[particleId];
  }
  // need to set this otherwise forces are zeros
  setLengthScaleToOne();
  //setSphericalLengthScale();
  cout << "DPM2D::setPolyRandomSoftParticles: particle packing fraction: " << getParticlePhi() << endl;
}

void DPM2D::setPolySizeDistribution(double calA0_, double polyDispersity) {
  calA0 = calA0_;
  double r1, r2, randNum, calA0temp;
  double numVertexInParticle, minVertexInParticle = numVertexPerParticle; // default
  numVertices = 0;
  // generate polydisperse number of vertices per particle
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    numVertexInParticle = floor(polyDispersity * numVertexPerParticle * randNum + numVertexPerParticle);
    if(numVertexInParticle < minVertexInParticle) {
      numVertexInParticle = minVertexInParticle;
    }
    // each particle has at least numVertexPerParticle vertices
    d_numVertexInParticleList[particleId] = numVertexInParticle;
    numVertices += numVertexInParticle;
  }
  cout << "DPM2D::setPolySizeDistribution: numVertices: " << numVertices << endl;
  cudaMemcpyToSymbol(d_numVertices, &(numVertices), sizeof(numVertices));
  setDimBlock(dimBlock); // recalculate dimGrid
  long* numVertexInParticleList = thrust::raw_pointer_cast(&d_numVertexInParticleList[0]);
  cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));

  // initialize everything else
  initParticleIdList();
  // we changed numVertices so we need to resize variables
  initVertexVariables(numVertices);
  initDynamicalVariables(numVertices);
  initNeighbors(numVertices);
  for (long particleId = 0; particleId < numParticles; particleId++) {
    numVertexInParticle = d_numVertexInParticleList[particleId];
    d_a0[particleId] = (numVertexInParticle / minVertexInParticle) * (numVertexInParticle / minVertexInParticle);
    calA0temp = calA0 * numVertexInParticle * tan(PI / numVertexInParticle) / PI;
    for (long vertexId = 0; vertexId < numVertexInParticle; vertexId++) {
      d_l0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * sqrt(PI * calA0temp * d_a0[particleId]) / numVertexInParticle;
  		d_theta0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * PI / numVertexInParticle;
  		d_rad[d_firstVertexInParticleId[particleId] + vertexId] = 0.5 * d_l0[d_firstVertexInParticleId[particleId] + vertexId];
      //cout << "vertexId: " << d_firstVertexInParticleId[particleId] + vertexId << " l0: " << d_l0[d_firstVertexInParticleId[particleId] + vertexId] << " rad: " << d_rad[d_firstVertexInParticleId[particleId] + vertexId] << endl;
    }
  }
}

void DPM2D::setSinusoidalRestAngles(double thetaA, double thetaK) {
  double thetaR;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    thetaR = 2. * PI / d_numVertexInParticleList[particleId];
    for (long vertexId = 0; vertexId < d_numVertexInParticleList[particleId]; vertexId++) {
      d_theta0[d_firstVertexInParticleId[particleId] + vertexId] = thetaA * thetaR * cos(thetaR * thetaK * vertexId);
    }
  }
}

// this works only for a square box
void DPM2D::setRandomParticles(double phi0, double extraRad_) {
  double boxLength = 1., scale = sqrt(getPreferredPhi() / phi0), extraRad = extraRad_;
  for (long dim = 0; dim < nDim; dim++) {
    d_boxSize[dim] = boxLength; // sqrt for 2d
  }
  double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
  cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  // extract random positions and radii
  double areaSum = 0;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_a0[particleId] /= (scale * scale);
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
    d_particleRad[particleId] = extraRad * sqrt((2. * d_a0[particleId]) / (d_numVertexInParticleList[particleId] * sin(2. * PI / d_numVertexInParticleList[particleId])));
    areaSum += PI * d_particleRad[particleId] * d_particleRad[particleId];
  }
  for(long vertexId = 0; vertexId < numVertices; vertexId++) {
    d_l0[vertexId] /= scale;
    d_rad[vertexId] /= scale;
  }
  // need to set this otherwise forces are zeros
  setLengthScale();
  cout << "DPM2D::setRandomParticles: particle packing fraction: " << getPreferredPhi() << " " << areaSum/(boxLength*boxLength) << endl;
}

void DPM2D::initVerticesOnParticles() {
  double rad;
  long particleId, numVertexInParticle;
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    particleId = d_particleIdList[vertexId];
    numVertexInParticle = d_numVertexInParticleList[particleId];
    rad = sqrt((2. * d_a0[particleId]) / (numVertexInParticle * sin(2. * PI / numVertexInParticle)));
		d_pos[vertexId * nDim] = rad * cos((2. * PI * vertexId) / numVertexInParticle) + d_particlePos[particleId * nDim] + 1e-02 * d_l0[vertexId] * drand48();
		d_pos[vertexId * nDim + 1] = rad * sin((2. * PI * vertexId) / numVertexInParticle) + d_particlePos[particleId * nDim + 1] + 1e-02 * d_l0[vertexId] * drand48();
  }
}

void DPM2D::scaleVertices(double scale) {
  thrust::host_vector<double> distance(nDim);
  calcParticlesPositions();
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    for (long dim = 0; dim < nDim; dim++) {
      distance[dim] = d_pos[vertexId * nDim + dim] - d_particlePos[d_particleIdList[vertexId]];
      d_pos[vertexId * nDim + dim] += (scale - 1.) * distance[dim];
    }
  }
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
  thrust::transform(d_area.begin(), d_area.end(), thrust::make_constant_iterator(scale * scale), d_area.begin(), thrust::multiplies<double>());
  thrust::transform(d_l0.begin(), d_l0.end(), thrust::make_constant_iterator(scale), d_l0.begin(), thrust::multiplies<double>());
  thrust::transform(d_rad.begin(), d_rad.end(), thrust::make_constant_iterator(scale), d_rad.begin(), thrust::multiplies<double>());
  distance.clear();
  setLengthScale();
}

void DPM2D::scaleParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
  setLengthScale();
}

void DPM2D::pressureScaleParticles(double pscale) {
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(pscale), d_particlePos.begin(), thrust::multiplies<double>());
  thrust::transform(d_boxSize.begin(), d_boxSize.end(), thrust::make_constant_iterator(pscale), d_boxSize.begin(), thrust::multiplies<double>());
}

void DPM2D::scaleSoftParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
  //setSphericalLengthScale();
}

void DPM2D::scaleParticleVelocity(double scale) {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), thrust::make_constant_iterator(scale), d_particleVel.begin(), thrust::multiplies<double>());
}

// translate vertices by particle displacement
void DPM2D::translateVertices() {
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), d_particleInitPos.begin(), d_particleDelta.begin(), thrust::minus<double>());
	double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double* pDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelTranslateVertices<<<dimGrid, dimBlock>>>(pDelta, pos);
}

// rotate vertices by particle angle change
void DPM2D::rotateVertices() {
	thrust::transform(d_particleAngle.begin(), d_particleAngle.end(), d_particleInitAngle.begin(), d_particleDeltaAngle.begin(), thrust::minus<double>());
	double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double* particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double* pDeltaAngle = thrust::raw_pointer_cast(&d_particleDeltaAngle[0]);
  kernelRotateVertices<<<dimGrid, dimBlock>>>(pDeltaAngle, particlePos, pos);
}

// compute particle angles from velocity
void DPM2D::computeParticleAngleFromVel() {
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
void DPM2D::setEnergyCosts(double ea_, double el_, double eb_, double ec_) {
  ea = ea_;
  el = el_;
  eb = eb_;
  ec = ec_;
  cudaMemcpyToSymbol(d_ea, &ea, sizeof(ea));
  cudaMemcpyToSymbol(d_el, &el, sizeof(el));
  cudaMemcpyToSymbol(d_eb, &eb, sizeof(eb));
  cudaMemcpyToSymbol(d_ec, &ec, sizeof(ec));
}

void DPM2D::setAttractionConstants(double l1_, double l2_) {
  l1 = l1_;
  l2 = l2_;
  cudaMemcpyToSymbol(d_l1, &l1, sizeof(l1));
  cudaMemcpyToSymbol(d_l2, &l2, sizeof(l2));
}

void DPM2D::setLJcutoff(double LJcutoff_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  LJecut = 4 * (1 / pow(LJcutoff, 12) - 1 / pow(LJcutoff, 6));
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
  //cout << "DPM2D::setLJcutoff - LJcutoff: " << LJcutoff << " LJecut: " << LJecut << endl;
}

double DPM2D::setTimeScale(double dt_) {
  double ta, tl, tb, tmin = 1e08;
  // compute typical time scale
  ta = rho0 / sqrt(ea);
  tl = (rho0 * d_l0[0]) / sqrt(ea * el); // TODO: replace values at 0 with averages
  tb = (rho0 * d_l0[0]) / sqrt(ea * eb); // TODO: replace values at 0 with averages
  // compute global time scale
  if (ta < tmin) tmin = ta;
  if (tl < tmin) tmin = tl;
  if (tb < tmin) tmin = tb;
  dt = tmin * dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

double DPM2D::setTimeStep(double dt_) {
  dt = dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

void DPM2D::calcForceEnergy() {
  thrust::fill(d_energy.begin(), d_energy.end(), double(0));
  calcParticlesShape();
  calcParticlesPositions();
  // shape variables
	const double *a0 = thrust::raw_pointer_cast(&d_a0[0]);
	const double *l0 = thrust::raw_pointer_cast(&d_l0[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
	const double *theta0 = thrust::raw_pointer_cast(&d_theta0[0]);
  // dynamical variables
  const double *area = thrust::raw_pointer_cast(&d_area[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	double *force = thrust::raw_pointer_cast(&d_force[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // compute shape force and vertex interaction
  kernelCalcShapeForceEnergy<<<dimGrid, dimBlock>>>(a0, area, particlePos, l0, theta0, pos, force, energy);
  //kernelCalcVertexInteraction<<<dimGrid, dimBlock>>>(rad, pos, force, energy);
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  kernelCalcVertexSegmentInteraction<<<dimGrid, dimBlock>>>(rad, pos, force, pEnergy);
}

void DPM2D::calcVertexForceAngAcc() {
  calcParticlesPositions();
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *force = thrust::raw_pointer_cast(&d_force[0]);
  double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // torque here is used for angular acceleration
  kernelCalcVertexForceAngAcc<<<dimGrid, dimBlock>>>(rad, pos, particlePos, force, torque, energy);
}

void DPM2D::calcRigidForceEnergy() {
  calcVertexForceAngAcc();
  // vertex variables
	const double *force = thrust::raw_pointer_cast(&d_force[0]);
  const double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	const double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // particle variables
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // sum force and torque over vertices of particle
  kernelCalcParticleForceAngAcc<<<dimGrid, dimBlock>>>(force, torque, energy, pForce, pTorque, pEnergy);
}

void DPM2D::calcVertexForceTorque() {
  calcParticlesPositions();
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *force = thrust::raw_pointer_cast(&d_force[0]);
  double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // torque here is used for angular acceleration
  kernelCalcVertexForceTorque<<<dimGrid, dimBlock>>>(rad, pos, particlePos, force, torque, energy);
}

void DPM2D::calcRigidForceTorque() {
  calcVertexForceTorque();
  // vertex variables
	const double *force = thrust::raw_pointer_cast(&d_force[0]);
  const double *torque = thrust::raw_pointer_cast(&d_torque[0]);
  // particle variables
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  // sum force and torque over vertices of particle
  kernelCalcParticleForceTorque<<<dimGrid, dimBlock>>>(force, torque, pForce, pTorque);
}

void DPM2D::calcStressTensor() {
  calcPerParticleStressTensor();
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *perPStress = thrust::raw_pointer_cast(&d_perParticleStress[0]);
	double *stress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcStressTensor<<<partDimGrid, dimBlock>>>(perPStress, stress);
}

void DPM2D::calcPerParticleStressTensor() {
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
	const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *perPStress = thrust::raw_pointer_cast(&d_perParticleStress[0]);
  kernelCalcPerParticleStressTensor<<<partDimGrid, dimBlock>>>(rad, pos, pPos, perPStress);
}

void DPM2D::calcNeighborForces() {
  thrust::host_vector<double> neighborForce;
  neighborForce.resize(numVertices * neighborListSize * nDim);
  thrust::fill(neighborForce.begin(), neighborForce.end(), 0);
	const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *neighforce = thrust::raw_pointer_cast(&neighborForce[0]);
  kernelCalcNeighborForces<<<dimGrid, dimBlock>>>(pos, rad, neighforce);
}

//************************* contacts and neighbors ***************************//
void DPM2D::calcParticleNeighbors() {
  long largestNeighbor = 8*nDim; // Guess
	do {
		//Make a contactList that is the right size
		neighborLimit = largestNeighbor;
		d_partNeighborList = thrust::device_vector<long>(numParticles * neighborLimit);
		//Prefill the contactList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		thrust::fill(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L);
		//Create device_pointers from thrust arrays
		double* pos = thrust::raw_pointer_cast(&d_pos[0]);
		double* rad = thrust::raw_pointer_cast(&d_rad[0]);
		long* pNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
		long* numPNeighbors = thrust::raw_pointer_cast(&d_numPartNeighbors[0]);
		kernelCalcParticleNeighbors<<<dimGrid, dimBlock>>>(pos, rad, neighborLimit, pNeighborList, numPNeighbors);
		//Calculate the maximum number of contacts
		largestNeighbor = thrust::reduce(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L, thrust::maximum<long>());
    //cout << "DPM2D::calcParticleNeighbors: largestNeighbor = " << largestNeighbor << endl;
	} while(neighborLimit < largestNeighbor); // If the guess was not good, do it again
}

void DPM2D::calcContacts(double gapSize) {
  long largestContact = 8*nDim; // Guess
	do {
		//Make a contactList that is the right size
		contactLimit = largestContact;
		d_contactList = thrust::device_vector<long>(numParticles * contactLimit);
		//Prefill the contactList with -1
		thrust::fill(d_contactList.begin(), d_contactList.end(), -1L);
		thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
		//Create device_pointers from thrust arrays
		const double* pos = thrust::raw_pointer_cast(&d_pos[0]);
		const double* rad = thrust::raw_pointer_cast(&d_rad[0]);
		long* contactList = thrust::raw_pointer_cast(&d_contactList[0]);
		long* numContacts = thrust::raw_pointer_cast(&d_numContacts[0]);
		kernelCalcContacts<<<dimGrid, dimBlock>>>(pos, rad, gapSize, contactLimit, contactList, numContacts);
		//Calculate the maximum number of contacts
		largestContact = thrust::reduce(d_numContacts.begin(), d_numContacts.end(), -1L, thrust::maximum<long>());
    //cout << "DPM2D::calcContacts: largestContact = " << largestContact << endl;
	} while(contactLimit < largestContact); // If the guess was not good, do it again
}

//Return normalized contact vectors between every pair of particles in contact
thrust::host_vector<long> DPM2D::getContactVectors(double gapSize) {
	//Calculate the set of contacts
	calcContacts(gapSize);
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

//*************************** vertex neighbors *******************************//
void DPM2D::calcNeighborList(double cutDistance) {
  thrust::fill(d_maxNeighborList.begin(), d_maxNeighborList.end(), 0);
	thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
  syncNeighborsToDevice();

  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double *rad = thrust::raw_pointer_cast(&d_rad[0]);

  kernelCalcNeighborList<<<dimGrid, dimBlock>>>(pos, rad, cutDistance);
  // compute maximum number of neighbors per particle
  maxNeighbors = thrust::reduce(d_maxNeighborList.begin(), d_maxNeighborList.end(), -1L, thrust::maximum<long>());
  syncNeighborsToDevice();
  //cout << "\n DPM2D::calcNeighborList: maxNeighbors = " << maxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( maxNeighbors > neighborListSize ) {
		neighborListSize = pow(2, ceil(std::log2(maxNeighbors)));
    //cout << "neighborListSize: " << neighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_neighborList.resize(numVertices * neighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
		syncNeighborsToDevice();
		kernelCalcNeighborList<<<dimGrid, dimBlock>>>(pos, rad, cutDistance);
	}
}

void DPM2D::syncNeighborsToDevice() {
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_neighborListSize, &neighborListSize, sizeof(neighborListSize));
	cudaMemcpyToSymbol(d_maxNeighbors, &maxNeighbors, sizeof(maxNeighbors));

	long* maxNeighborList = thrust::raw_pointer_cast(&d_maxNeighborList[0]);
	cudaMemcpyToSymbol(d_maxNeighborListPtr, &maxNeighborList, sizeof(maxNeighborList));

	long* neighborList = thrust::raw_pointer_cast(&d_neighborList[0]);
	cudaMemcpyToSymbol(d_neighborListPtr, &neighborList, sizeof(neighborList));
}

//************************* particle functions *******************************//
void DPM2D::calcParticleForceEnergy() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void DPM2D::calcParticleWallForceEnergy() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteractionFixedBoundary<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
  kernelCalcParticleWallInteraction<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void DPM2D::calcParticleSidesForceEnergy() {
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteractionFixedSides<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
  kernelCalcParticleSidesInteraction<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void DPM2D::calcParticleForceEnergyRA() { // Repulsive and Attractive
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteractionRA<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

void DPM2D::calcParticleForceEnergyLJ() { // Repulsive and Attractive
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteractionLJ<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

 void DPM2D::makeExternalParticleForce(double externalForce) {
   // extract +-1 random forces
   d_particleDelta.resize(numParticles);
   thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
   thrust::counting_iterator<long> index_sequence_begin(lrand48());
   thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleDelta.begin(), randInt(0,1));
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(2), d_particleDelta.begin(), thrust::multiplies<double>());
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(1), d_particleDelta.begin(), thrust::minus<double>());
   thrust::transform(d_particleDelta.begin(), d_particleDelta.end(), thrust::make_constant_iterator(externalForce), d_particleDelta.begin(), thrust::multiplies<double>());
 }

 void DPM2D::addExternalParticleForce() {
   long p_nDim(nDim);
   auto r = thrust::counting_iterator<long>(0);
 	 double *pDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
 	 double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);

   auto addExternalForce = [=] __device__ (long particleId) {
     pForce[particleId * p_nDim] += pDelta[particleId];
   };

   thrust::for_each(r, r + numParticles, addExternalForce);
 }

 thrust::host_vector<double> DPM2D::getExternalParticleForce() {
   // return signed external forces
   thrust::host_vector<double> particleExternalForce;
   particleExternalForce = d_particleDelta;
   return particleExternalForce;
 }

 // return the sum of force magnitudes
 double DPM2D::getParticleTotalForceMagnitude() {
   thrust::device_vector<double> forceSquared(d_force.size());
   // compute squared velocities
   thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
   // sum squares
   double totalForceMagnitude = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(0), thrust::plus<double>()) / (numParticles * nDim));
   forceSquared.clear();
   return totalForceMagnitude;
 }

double DPM2D::getParticleMaxUnbalancedForce() {
  thrust::device_vector<double> forceSquared(d_particleForce.size());
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  double maxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
  return maxUnbalancedForce;
}

double DPM2D::getRigidMaxUnbalancedForce() {
  //calcRigidForceEnergy();
  thrust::device_vector<double> forceSquared(d_particleForce.size());
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  double particleMaxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
	forceSquared.resize(d_particleTorque.size());
	thrust::transform(d_particleTorque.begin(), d_particleTorque.end(), forceSquared.begin(), square());
	double particleMaxUnbalancedTorque = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
	return std::max(particleMaxUnbalancedForce, particleMaxUnbalancedTorque);
}

void DPM2D::calcParticleStressTensor() {
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *pStress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcParticleStressTensor<<<partDimGrid, dimBlock>>>(pRad, pPos, pStress);
}

double DPM2D::getParticleVirialPressure() {
   calcParticleStressTensor();
	 double totalStress = 0, volume = 1;
	 for (long dim = 0; dim < nDim; dim++) {
		 totalStress += d_stress[dim * nDim + dim];
     volume *= d_boxSize[dim];
	 }
	 return totalStress / (nDim * volume);
	 //return totalStress;
}

double DPM2D::getParticleDynamicalPressure() {
  double volume = 1;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
  }
  return getParticleTemperature() * numParticles / volume;
}

double DPM2D::getParticleWallPressure() {
	 double wallWork = 0, volume = 1;
	 for (long dim = 0; dim < nDim; dim++) {
     volume *= d_boxSize[dim];
	 }
   const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
   const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
   kernelCalcParticleWallPressure<<<partDimGrid, dimBlock>>>(pRad, pPos, wallWork);
	 return wallWork / (nDim * volume);
	 //return totalStress;
}

double DPM2D::getParticleActivePressure(double driving) {
  double activeWork = 0, volume = 1;
  for (long dim = 0; dim < nDim; dim++) {
    volume *= d_boxSize[dim];
  }
	const double *pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  kernelCalcParticleActivePressure<<<partDimGrid, dimBlock>>>(pAngle, pPos, driving, activeWork);

  return activeWork / (nDim * volume);
}

double DPM2D::getParticleTotalPressure(double driving) {
  return getParticleDynamicalPressure() + getParticleActivePressure(driving);
}

double DPM2D::getParticleEnergy() {
  return thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
}

double DPM2D::getParticleKineticEnergy() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  // compute squared velocities
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  // sum squares
  //cout << "vel squared: " << velSquared[0] << " " << velSquared[1] << " " << thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>()) << endl;
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
}

double DPM2D::getParticleTemperature() {
  double ekin = getParticleKineticEnergy();
  return 2 * ekin / (numParticles * nDim);
}

double DPM2D::getMassiveTemperature(long firstIndex, double mass) {
  // temperature computed from the massive particles which are set to be the first #
  thrust::device_vector<double> velSquared(firstIndex * nDim);
  // compute squared velocities
  thrust::transform(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, velSquared.begin(), square());
  return mass * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>()) / (firstIndex * nDim);
}

double DPM2D::getParticleDrift() {
  return thrust::reduce(d_particlePos.begin(), d_particlePos.end(), double(0), thrust::plus<double>()) / (numParticles * nDim);
}

thrust::host_vector<long> DPM2D::getParticleNeighbors() {
  thrust::host_vector<long> partNeighborListFromDevice;
  partNeighborListFromDevice = d_partNeighborList;
  return partNeighborListFromDevice;
}

void DPM2D::calcParticleNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleNeighborList<<<partDimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  partMaxNeighbors = thrust::reduce(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncParticleNeighborsToDevice();
  //cout << "DPM2D::calcParticleNeighborList: maxNeighbors: " << partMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( partMaxNeighbors > partNeighborListSize ) {
		partNeighborListSize = pow(2, ceil(std::log2(partMaxNeighbors)));
    //cout << "DPM2D::calcParticleNeighborList: neighborListSize: " << neighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_partNeighborList.resize(numParticles * partNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		syncParticleNeighborsToDevice();
		kernelCalcParticleNeighborList<<<partDimGrid, dimBlock>>>(pPos, pRad, cutDistance);
	}
}

void DPM2D::syncParticleNeighborsToDevice() {
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_partNeighborListSize, &partNeighborListSize, sizeof(partNeighborListSize));
	cudaMemcpyToSymbol(d_partMaxNeighbors, &partMaxNeighbors, sizeof(partMaxNeighbors));

	long* partMaxNeighborList = thrust::raw_pointer_cast(&d_partMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_partMaxNeighborListPtr, &partMaxNeighborList, sizeof(partMaxNeighborList));

	long* partNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
	cudaMemcpyToSymbol(d_partNeighborListPtr, &partNeighborList, sizeof(partNeighborList));
}

void DPM2D::calcParticleWallNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleWallNeighborList<<<partDimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  partMaxNeighbors = thrust::reduce(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncParticleNeighborsToDevice();
  //cout << "DPM2D::calcParticleNeighborList: maxNeighbors: " << partMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( partMaxNeighbors > partNeighborListSize ) {
		partNeighborListSize = pow(2, ceil(std::log2(partMaxNeighbors)));
    //cout << "DPM2D::calcParticleNeighborList: neighborListSize: " << neighborListSize << endl;
		//Now create the actual storage and then put the neighbors in it.
		d_partNeighborList.resize(numParticles * partNeighborListSize);
		//Pre-fill the neighborList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		syncParticleNeighborsToDevice();
		kernelCalcParticleNeighborList<<<partDimGrid, dimBlock>>>(pPos, pRad, cutDistance);
	}
}

void DPM2D::calcParticleContacts(double gapSize) {
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
    //cout << "DPM2D::calcParticleContacts: largestContact = " << largestContact << endl;
	} while(contactLimit < largestContact); // If the guess was not good, do it again
}

//************************** minimizer functions *****************************//
void DPM2D::initFIRE(std::vector<double> &FIREparams, long minStep_, long numStep_, long numDOF_) {
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
    cout << "DPM2D::initFIRE: wrong number of FIRE parameters, must be 7" << endl;
  }
}

void DPM2D::setParticleMassFIRE() {
  //this->fire_->setParticleMass();
  this->fire_->d_mass.resize(numParticles * nDim);
	for (long particleId = 0; particleId < numParticles; particleId++) {
		for (long dim = 0; dim < nDim; dim++) {
			this->fire_->d_mass[particleId * nDim + dim] = PI / (d_particleRad[particleId] * d_particleRad[particleId]);
		}
	}
}

void DPM2D::setTimeStepFIRE(double timeStep_) {
  this->fire_->setFIRETimeStep(timeStep_);
}


void DPM2D::particleFIRELoop() {
  this->fire_->minimizerParticleLoop();
}

void DPM2D::vertexFIRELoop() {
  this->fire_->minimizerVertexLoop();
}

void DPM2D::initRigidFIRE(std::vector<double> &FIREparams, long minStep_, long numStep_, long numDOF_, double cutDist_) {
  initFIRE(FIREparams, minStep_, numStep_, numDOF_);
  initDeltaVariables(getNumVertices(), getNumParticles());
  initRotationalVariables(getNumVertices(), getNumParticles());
  this->fire_->cutDistance = cutDist_;
}

void DPM2D::rigidFIRELoop() {
  this->fire_->minimizerRigidLoop();
}

//********************* deformable particles integrators *********************//
void DPM2D::initLangevin(double Temp, double gamma, bool readState) {
  this->sim_ = new Langevin2(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlesPositions();
  d_particleInitPos = getParticlePositions();
  //cout << "DPM2D::initLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  if(readState == false) {
    //this->sim_->injectKineticEnergy();
    thrust::fill(d_vel.begin(), d_vel.end(), double(0));
  }
  cout << "DPM2D::initLangevin:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::langevinLoop() {
  this->sim_->integrate();
}

void DPM2D::initActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new ActiveLangevin(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlesPositions();
  d_particleInitPos = getParticlePositions();
  //cout << "DPM2D::initActiveLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  if(readState == false) {
    //this->sim_->injectKineticEnergy();
    thrust::fill(d_vel.begin(), d_vel.end(), double(0));
  }
  cout << "DPM2D::initActiveLangevin:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeLangevinLoop() {
  this->sim_->integrate();
}

void DPM2D::initNVE(double Temp, bool readState) {
  this->sim_ = new NVE(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->noiseVar = sqrt(2. * Temp);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initNVE:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::NVELoop() {
  this->sim_->integrate();
}

void DPM2D::initBrownian(double Temp, double gamma, bool readState) {
  this->sim_ = new Brownian(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  cout << "DPM2D::initBrownian:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::brownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initActiveBrownian(double Dr, double driving, bool readState) {
  this->sim_ = new ActiveBrownian(this, SimConfig(0, Dr, driving, 0));
  this->sim_->d_rand.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //thrust::counting_iterator<long> index_sequence_begin(drand48());
    //thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleAngle.begin(), randNum(0.f, 2.f * PI));
  }
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  cout << "DPM2D::initActiveBrownian:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeBrownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initActiveBrownianDampedL0(double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new ActiveBrownianDampedL0(this, SimConfig(0, Dr, driving, 0));
  this->sim_->lcoeff1 = exp(-gamma * dt);
  this->sim_->d_rand.resize(numParticles);
  d_l0Vel.resize(numVertices);
  thrust::fill(d_l0Vel.begin(), d_l0Vel.end(), double(0));
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //thrust::counting_iterator<long> index_sequence_begin(drand48());
    //thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleAngle.begin(), randNum(0.f, 2.f * PI));
  }
  cout << "DPM2D::initActiveBrownian:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeBrownianDampedL0Loop() {
  this->sim_->integrate();
}

//************************* soft particle simulators *************************//
void DPM2D::computeParticleDrift() {
  thrust::fill(d_delta.begin(), d_delta.end(), double(0));
  double *velSum = thrust::raw_pointer_cast(&d_delta[0]);
  const double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  kernelSumParticleVelocity<<<partDimGrid, dimBlock>>>(pVel, velSum);
}

void DPM2D::conserveParticleMomentum() {
  double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double *velSum = thrust::raw_pointer_cast(&d_delta[0]);
  kernelSubtractParticleDrift<<<partDimGrid, dimBlock>>>(pVel, velSum);
}

void DPM2D::initSoftParticleLangevin(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevin2(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  //d_delta.resize(nDim);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    //cout << "DPM2D::initSoftParticleLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "DPM2D::initSoftParticleLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleLangevinLoop() {
  this->sim_->integrate();
  //computeParticleDrift();
  //conserveParticleMomentum();
  //computeParticleDrift();
  //cout << "velSum: " << thrust::reduce(d_particleVel.begin(), d_particleVel.end(), double(0), thrust::plus<double>()) << endl;
}

void DPM2D::initSoftParticleLangevinFixedBoundary(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevinFixedBoundary(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    //cout << "DPM2D::initSoftParticleLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "DPM2D::initSoftParticleLangevinFixedBoundary:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleLangevinFixedBoundaryLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleNVE(double Temp, bool readState) {
  this->sim_ = new SoftParticleNVE(this, SimConfig(Temp, 0, 0, 0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initSoftParticleNVE:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleNVELoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleNVERA(double Temp, bool readState) {
  this->sim_ = new SoftParticleNVERA(this, SimConfig(Temp, 0, 0, 0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initSoftParticleNVERA:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleNVERALoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleNVEFixedBoundary(double Temp, bool readState) {
  this->sim_ = new SoftParticleNVEFixedBoundary(this, SimConfig(Temp, 0, 0, 0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initSoftParticleNVEFixedBoundary:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleNVEFixedBoundaryLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleActiveNVEFixedBoundary(double Temp, double Dr, double driving, bool readState) {
  this->sim_ = new SoftParticleActiveNVEFixedBoundary(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->d_pActiveAngle.resize(numParticles);
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "DPM2D::initSoftParticleActiveNVEFixedBoundary:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleActiveNVEFixedBoundaryLoop() {
  d_particlePreviousPos = getParticlePositions();
  this->sim_->integrate();
}

void DPM2D::initSoftParticleLangevinRA(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevin2RA(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    //cout << "DPM2D::initSoftParticleLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "DPM2D::initSoftParticleLangevinRA:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleLangevinRALoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleLangevinLJ(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLangevin2LJ(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    //cout << "DPM2D::initSoftParticleLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "DPM2D::initSoftParticleLangevinLJ:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleLangevinLJLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftLangevinSubSet(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel) {
  this->sim_ = new SoftLangevinSubSet(this, SimConfig(Temp, 0, 0, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  if(zeroOutMassiveVel == true) {
    thrust::fill(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, double(0));
  }
  cout << "DPM2D::initSoftLangevinSubSet:: current temperature: " << setprecision(12) << getParticleTemperature() << " mass: " << this->sim_->mass << endl;
}

void DPM2D::softLangevinSubSetLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleLExtField(double Temp, double gamma, bool readState) {
  this->sim_ = new SoftParticleLExtField(this, SimConfig(Temp, 0, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initSoftParticleLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleLExtFieldLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveLangevin(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //cout << "DPM2D::initSoftParticleActiveLangevin:: damping coefficients: " << this->sim_->lcoeff1 << " " << this->sim_->lcoeff2 << " " << this->sim_->lcoeff3 << endl;
  }
  cout << "DPM2D::initSoftParticleActiveLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleActiveLangevinLoop() {
  //d_particlePreviousPos = getParticlePositions();
  this->sim_->integrate();
}

void DPM2D::initSoftParticleActiveLangevinFixedBoundary(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveLangevinFixedBoundary(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "DPM2D::initSoftParticleActiveLangevinFixedBoundary:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleActiveLangevinFixedBoundaryLoop() {
  this->sim_->integrate();

}

void DPM2D::initSoftParticleActiveLangevinFixedSides(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleActiveLangevinFixedSides(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "DPM2D::initSoftParticleActiveLangevinFixedSides:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleActiveLangevinFixedSidesLoop() {
  this->sim_->integrate();

}

void DPM2D::initSoftALSubSet(double Temp, double Dr, double driving, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel) {
  this->sim_ = new SoftALSubSet(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  if(zeroOutMassiveVel == true) {
    thrust::fill(d_particleVel.begin(), d_particleVel.begin() + firstIndex * nDim, double(0));
  }
  cout << "DPM2D::initSoftALSubSet:: current temperature: " << setprecision(12) << getParticleTemperature() << " mass: " << this->sim_->mass << endl;
}

void DPM2D::softALSubSetLoop() {
  this->sim_->integrate();
}

void DPM2D::initSoftParticleALExtField(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new SoftParticleALExtField(this, SimConfig(Temp, Dr, driving, 0));
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
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particleInitPos = getParticlePositions();
  d_particlePreviousPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  d_particlePreviousPos = getParticlePositions();
  d_particleDelta.resize(numParticles * nDim);
  d_particleDisp.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "DPM2D::initSoftParticleALExtField:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::softParticleALExtFieldLoop() {
  this->sim_->integrate();
}


//************************* rigid particle simulators ************************//
void DPM2D::initRigidBrownian(double Temp, double cutDistance, bool readState) {
  this->sim_ = new RigidBrownian(this, SimConfig(Temp, 0, 0, cutDistance));
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  if(readState == false) {
      this->sim_->injectKineticEnergy();
  }
  initDeltaVariables(getNumVertices(), getNumParticles());
  initRotationalVariables(getNumVertices(), getNumParticles());
  d_initialPos = getVertexPositions();
  calcParticlesPositions();
  d_particleInitPos = getParticlePositions();
  cout << "DPM2D::initRigidBrownian:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::rigidBrownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initRigidRotActiveBrownian(double Dr, double driving, double cutDistance, bool readState) {
  this->sim_ = new RigidRotActiveBrownian(this, SimConfig(0, Dr, driving, cutDistance));
  this->sim_->d_rand.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //thrust::counting_iterator<long> index_sequence_begin(drand48());
    //thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleAngle.begin(), randNum(0.f, 2.f * PI));
  }
  initDeltaVariables(getNumVertices(), getNumParticles());
  initRotationalVariables(getNumVertices(), getNumParticles());
  d_initialPos = getVertexPositions();
  calcParticlesPositions();
  d_particleInitPos = getParticlePositions();
  cout << "DPM2D::initRigidRotActiveBrownian:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::rigidRotActiveBrownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initRigidActiveBrownian(double Dr, double driving, double cutDistance, bool readState) {
  this->sim_ = new RigidActiveBrownian(this, SimConfig(0, Dr, driving, cutDistance));
  this->sim_->d_rand.resize(numParticles);
  this->sim_->d_pActiveAngle.resize(numParticles);
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
    //thrust::counting_iterator<long> index_sequence_begin(drand48());
    //thrust::transform(index_sequence_begin, index_sequence_begin + numParticles, d_particleAngle.begin(), randNum(0.f, 2.f * PI));
  }
  initDeltaVariables(getNumVertices(), getNumParticles());
  initRotationalVariables(getNumVertices(), getNumParticles());
  d_initialPos = getVertexPositions();
  calcParticlesPositions();
  d_particleInitPos = getParticlePositions();
  cout << "DPM2D::initRigidActiveBrownian:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::rigidActiveBrownianLoop() {
  this->sim_->integrate();
}
