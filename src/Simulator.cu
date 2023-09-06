//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//
// DEFINITION OF INTEGRATION FUNCTIONS

#include "../include/Simulator.h"
#include "../include/defs.h"
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

using namespace std;
// vertex updates
__global__ void kernelUpdateVertexPos(double* pos, const double* vel, const double timeStep);
__global__ void kernelUpdateVertexVel(double* vel, const double* force, const double timeStep);
//__global__ void kernelExtractThermalVertexVel(double* vel, const double* r1, const double* r2, const double amplitude);
__global__ void kernelUpdateBrownianVertexVel(double* vel, const double* force, double* thermalVel, const double mobility);
__global__ void kernelUpdateActiveVertexVel(double* vel, const double* force, double* pAngle, const double driving, const double mobility);
// rigid updates
__global__ void kernelUpdateRigidPos(double* pPos, const double* pVel, double* pAngle, const double* pAngvel, const double timeStep);
__global__ void kernelUpdateRigidBrownianVel(double* pVel, const double* pForce, double* pAngvel, const double* pTorque, double* thermalVel, const double mobility);
__global__ void kernelUpdateRigidActiveVel(double* pVel, const double* pForce, double* pActiveAngle, double* pAngvel, const double* pTorque, const double driving, const double mobility);
// particle updates
__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep);
__global__ void kernelUpdateParticleVel(double* pVel, const double* pForce, const double timeStep);
//__global__ void kernelExtractThermalParticleVel(double* pVel, const double* r1, const double* r2, const double amplitude);
__global__ void kernelUpdateBrownianParticleVel(double* pVel, const double* pForce, double* thermalVel, const double mobility);
__global__ void kernelUpdateActiveParticleVel(double* pVel, const double* pForce, double* pAngle, const double driving, const double mobility);
// momentum conservation
__global__ void kernelConserveVertexMomentum(double* vel);
__global__ void kernelConserveParticleMomentum(double* pVel);
__global__ void kernelConserveSubSetMomentum(double* pVel, const long firstIndex);


//********************************** langevin ********************************//
void Langevin::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(0.5 * dpm_->dt);
  updateThermalVel();
  updatePosition(0.5 * dpm_->dt);
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  conserveMomentum();
}

void Langevin::injectKineticEnergy() {
  // generate random numbers between 0 and Tscale for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  long s_nDim(dpm_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinInjectThermalVel = [=] __device__ (long vertexId) {
    long particleId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinInjectThermalVel);
  kernelConserveVertexMomentum<<<1, dpm_->dimBlock>>>(vel);
}

void Langevin::updatePosition(double timeStep) {
	double* pos = thrust::raw_pointer_cast(&dpm_->d_pos[0]);
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelUpdateVertexPos<<<dpm_->dimGrid, dpm_->dimBlock>>>(pos, vel, timeStep);
}

void Langevin::updateVelocity(double timeStep) {
	double* vel = thrust::raw_pointer_cast(&dpm_->d_vel[0]);
	const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateVertexPos<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, timeStep);
}

void Langevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateThermalVel = [=] __device__ (long vertexId) {
    long particleId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = s_lcoeff1 * vel[vertexId * s_nDim + dim] + s_lcoeff2 * thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateThermalVel);
}

void Langevin::conserveMomentum() {
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelConserveVertexMomentum<<<1, dpm_->dimBlock>>>(vel);
}

//***************************** vertex langevin2 *****************************//
void Langevin2::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void Langevin2::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);
}

void Langevin2::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
	const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateVertexVel = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vId * s_nDim + dim] += s_dt * (force[vId * s_nDim + dim] - vel[vId * s_nDim + dim] * s_gamma);
      vel[vId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (force[vId * s_nDim + dim] - vel[vId * s_nDim + dim] * s_gamma) / s_gamma;
      vel[vId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateVertexVel);
}

void Langevin2::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pos = thrust::raw_pointer_cast(&(dpm_->d_pos[0]));
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateVertexPos = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pos[vId * s_nDim + dim] += s_dt * vel[vId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateVertexPos);
}


//***************************** active langevin ******************************//
void ActiveLangevin::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void ActiveLangevin::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);

  // generate active forces
  double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  double s_driving(config.driving);
  auto s = thrust::counting_iterator<long>(0);
  const double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto addActiveParticleForceToVertex = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    pAngle[pId] += amplitude * pActiveAngle[pId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      force[vId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId]));
    }
  };

  thrust::for_each(s, s + dpm_->numVertices, addActiveParticleForceToVertex);
}


//************************************ NVE ***********************************//
void NVE::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

//******************************** brownian **********************************//
void Brownian::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  conserveMomentum();
}

void Brownian::updateVelocity(double timeStep) {
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  // update vertex velocity
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  kernelUpdateBrownianVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, thermalVel, mobility);
}

//**************************** active brownian *******************************//
void ActiveBrownian::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  conserveMomentum();
}

void ActiveBrownian::updateVelocity(double timeStep) {
  double amplitude = sqrt(2. * config.Dr);
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  auto r = thrust::counting_iterator<long>(0);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);

  auto updateAngle = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * rand[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateAngle);
  // update vertex velocity with overdamped active dynamics
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateActiveVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, pAngle, config.driving, mobility);
}

//******************* active brownian with damping on l0 *********************//
void ActiveBrownianDampedL0::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcForceEnergy();
  conserveMomentum();
}

void ActiveBrownianDampedL0::updatePosition(double timeStep) {
	double* pos = thrust::raw_pointer_cast(&dpm_->d_pos[0]);
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelUpdateVertexPos<<<dpm_->dimGrid, dpm_->dimBlock>>>(pos, vel, timeStep);
  // update rest length
  auto r = thrust::counting_iterator<long>(0);
  double s_dt(timeStep);
  double s_kl = 1.;
  double s_lcoeff1(lcoeff1);
  const double *length = thrust::raw_pointer_cast(&(dpm_->d_length[0]));
  double *l0 = thrust::raw_pointer_cast(&(dpm_->d_l0[0]));
  double *l0Vel = thrust::raw_pointer_cast(&(dpm_->d_l0Vel[0]));

  auto firstUpdateRestLength = [=] __device__ (long vertexId) {
    l0Vel[vertexId] += s_kl * (length[vertexId] - l0[vertexId]) * s_dt * 0.5;
    l0[vertexId] += l0Vel[vertexId] * s_dt * 0.5;
    l0Vel[vertexId] = s_lcoeff1 * l0Vel[vertexId];
  };

  thrust::for_each(r, r + dpm_->numVertices, firstUpdateRestLength);
}

void ActiveBrownianDampedL0::updateVelocity(double timeStep) {
  double amplitude = sqrt(2. * config.Dr);
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  auto r = thrust::counting_iterator<long>(0);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);

  auto updateAngle = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * rand[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateAngle);
  // update vertex velocity with overdamped active dynamics
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateActiveVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, pAngle, config.driving, mobility);
  // update rest length
  auto s = thrust::counting_iterator<long>(0);
  double s_dt(timeStep);
  double s_kl = 1.;
  const double *length = thrust::raw_pointer_cast(&(dpm_->d_length[0]));
  double *l0 = thrust::raw_pointer_cast(&(dpm_->d_l0[0]));
  double *l0Vel = thrust::raw_pointer_cast(&(dpm_->d_l0Vel[0]));

  auto secondUpdateRestLength = [=] __device__ (long vertexId) {
		l0[vertexId] += l0Vel[vertexId] * s_dt * 0.5;
		l0Vel[vertexId] += s_kl * (length[vertexId] - l0[vertexId]) * s_dt * 0.5;
  };

  thrust::for_each(r, r + dpm_->numVertices, secondUpdateRestLength);
}

//************************* soft particle langevin ***************************//
void SoftParticleLangevin::integrate() {
  updateVelocity(0.5*dpm_->dt);
  updatePosition(0.5*dpm_->dt);
  updateThermalVel();
  updatePosition(0.5*dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  conserveMomentum();
}

void SoftParticleLangevin::injectKineticEnergy() {
  double amplitude(sqrt(config.Tinject));
  // generate random numbers between 0 and noiseVar for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, dpm_->d_particleVel.begin(), gaussNum(0.f,amplitude));
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}

void SoftParticleLangevin::updatePosition(double timeStep) {
	double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelUpdateParticlePos<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pPos, pVel, timeStep);
}

void SoftParticleLangevin::updateVelocity(double timeStep) {
  double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
  kernelUpdateParticleVel<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pVel, pForce, timeStep);
}

void SoftParticleLangevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalVel = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[particleId * s_nDim + dim] = s_lcoeff1 * pVel[particleId * s_nDim + dim] + s_lcoeff2 * thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalVel);
}

void SoftParticleLangevin::conserveMomentum() {
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}

//**************************** soft particle nve *****************************//
void SoftParticleNVE::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

//*********************** attractive soft particle nve ***********************//
void SoftParticleNVERA::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergyRA();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

//********************* fixed boundary soft particle nve *********************//
void SoftParticleNVEFixedBoundary::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleWallForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

//********************* fixed boundary soft particle nve *********************//
void SoftParticleActiveNVEFixedBoundary::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleWallForceEnergy();
  updateThermalVel();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

void SoftParticleActiveNVEFixedBoundary::updateThermalVel() {
  auto r = thrust::counting_iterator<long>(0);
  long s_nDim(dpm_->nDim);
  double s_driving(config.driving);
  double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  auto nveUpdateActiveNoise = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * pActiveAngle[particleId];
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      pForce[particleId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId]));
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, nveUpdateActiveNoise);
}

//************************* soft particle langevin ***************************//
void SoftParticleLangevin2::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void SoftParticleLangevin2::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);
}

void SoftParticleLangevin2::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) * s_gamma;
      pVel[pId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateParticleVel);
  //kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}

void SoftParticleLangevin2::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));

  auto langevinUpdateParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateParticlePos);
}

//************************ soft particle langevinRA **************************//
void SoftParticleLangevin2RA::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergyRA(); // this is the only difference
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

//*********************** soft particle Lennard-Jones ************************//
void SoftParticleLangevin2LJ::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergyLJ(); // this is the only difference
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

//****************** fixed boundary soft particle langevin *******************//
void SoftParticleLangevinFixedBoundary::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleWallForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

//************** soft particle langevin with massive particles ***************//
void SoftLangevinSubSet::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  conserveMomentum();
}

void SoftLangevinSubSet::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>();
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r + firstIndex, r + dpm_->numParticles, langevinUpdateThermalNoise);
}

void SoftLangevinSubSet::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) * s_gamma;
      pVel[pId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r + firstIndex, r + dpm_->numParticles, langevinUpdateParticleVel);
  // update massive particles with normal langevin velocity update
  double s_mass(mass);
  auto s = thrust::counting_iterator<long>(0);
  auto updateMassiveParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * pForce[pId * s_nDim + dim] / s_mass;
    }
  };

  thrust::for_each(s, s + firstIndex, updateMassiveParticleVel);
}

void SoftLangevinSubSet::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));

  auto langevinUpdateParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r + firstIndex, r + dpm_->numParticles, langevinUpdateParticlePos);
  // update massive particles with normal langevin position update
  auto s = thrust::counting_iterator<long>(0);
  auto updateMassiveParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(s, s + firstIndex, updateMassiveParticlePos);
}

void SoftLangevinSubSet::conserveMomentum() {
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveSubSetMomentum<<<1, dpm_->dimBlock>>>(pVel, firstIndex);
  //kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}

//************************* soft particle langevin ***************************//
void SoftParticleLExtField::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  dpm_->addExternalParticleForce();
  updateVelocity(0.5*dpm_->dt);
  conserveMomentum();
}

//********************** soft particle active langevin ***********************//
void SoftParticleActiveLangevin::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void SoftParticleActiveLangevin::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  // generate active forces
  double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  double s_driving(config.driving);
  const double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * pActiveAngle[particleId];
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
      pForce[particleId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId]));
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);
  // generate active forces
  //double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  //thrust::counting_iterator<long> index_sequence_begin(lrand48());
  //thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  //double s_driving(config.driving);
  //auto s = thrust::counting_iterator<long>(0);
  //const double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  //double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	//double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  //auto addActiveParticleForce = [=] __device__ (long particleId) {
  //  pAngle[particleId] += amplitude * pActiveAngle[particleId];
  //  #pragma unroll (MAXDIM)
	//	for (long dim = 0; dim < s_nDim; dim++) {
  //    pForce[particleId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId]));
  //  }
  //};

  //thrust::for_each(s, s + dpm_->numParticles, addActiveParticleForce);
}

//************** fixed boundary soft particle active langevin ****************//
void SoftParticleActiveLangevinFixedBoundary::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleWallForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

//*************** fixed x sides soft particle active langevin ****************//
void SoftParticleActiveLangevinFixedSides::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleSidesForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}


//*********** soft particle active langevin with massive particles ***********//
void SoftALSubSet::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  conserveMomentum();
}

void SoftALSubSet::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>();
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r + firstIndex, r + dpm_->numParticles, langevinUpdateThermalNoise);
  // generate active forces
  double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  double s_driving(config.driving);
  auto s = thrust::counting_iterator<long>(0);
  const double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

  auto addActiveParticleForce = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * pActiveAngle[particleId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pForce[particleId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId]));
    }
  };

  thrust::for_each(s + firstIndex, s + dpm_->numParticles, addActiveParticleForce);
}

//********************** soft particle active langevin ***********************//
void SoftParticleALExtField::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->calcParticleForceEnergy();
  dpm_->addExternalParticleForce();
  updateVelocity(0.5*dpm_->dt);
  conserveMomentum();
}


//****************************** rigid brownian ******************************//
void RigidBrownian::integrate() {
	dpm_->d_particleInitPos = dpm_->getParticlePositions();
	dpm_->d_particleInitAngle = dpm_->getParticleAngles();
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->translateVertices();
	dpm_->rotateVertices();
	dpm_->calcNeighborList(config.cutDist);
  dpm_->calcRigidForceTorque();
  conserveMomentum();
}

void RigidBrownian::updatePosition(double timeStep) {
	double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
  double* pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
  kernelUpdateRigidPos<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pPos, pVel, pAngle, pAngvel, timeStep);
}

void RigidBrownian::updateVelocity(double timeStep) {
  double mobility = 1/gamma;
  double amplitude = sqrt(2. * config.Tinject * gamma / timeStep);
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,amplitude));
  // update vertex velocity
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
  const double *pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double* pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));
  kernelUpdateRigidBrownianVel<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pVel, pForce, pAngvel, pTorque, thermalVel, mobility);
}

//************************** rigid active brownian ***************************//
// active angle and particle rotation angle are decoupled
void RigidActiveBrownian::integrate() {
	dpm_->d_particleInitPos = dpm_->getParticlePositions();
	dpm_->d_particleInitAngle = dpm_->getParticleAngles();
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->translateVertices();
	dpm_->rotateVertices();
	dpm_->calcNeighborList(config.cutDist);
  dpm_->calcRigidForceTorque();
  conserveMomentum();
}

void RigidActiveBrownian::updateVelocity(double timeStep) {
  double mobility = 1/gamma;
  double amplitude = sqrt(2. * config.Dr);
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  auto r = thrust::counting_iterator<long>(0);
  double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);

  auto updateActiveAngle = [=] __device__ (long particleId) {
    pActiveAngle[particleId] += amplitude * rand[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateActiveAngle);
  // update particle velocity with overdamped active dynamics
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
  const double *pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double* pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));
  kernelUpdateRigidActiveVel<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pVel, pForce, pActiveAngle, pAngvel, pTorque, config.driving, mobility);
}

//******************** rigid rotational active brownian **********************//
// active angle and particle rotation angle are coupled
void RigidRotActiveBrownian::integrate() {
	dpm_->d_particleInitPos = dpm_->getParticlePositions();
	dpm_->d_particleInitAngle = dpm_->getParticleAngles();
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->translateVertices();
	dpm_->rotateVertices();
	dpm_->calcNeighborList(config.cutDist);
  dpm_->calcRigidForceTorque();
  conserveMomentum();
}

void RigidRotActiveBrownian::updateVelocity(double timeStep) {
  double mobility = 1/gamma;
  double amplitude = sqrt(2. * config.Dr);
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
	const double* pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));

  auto updateRigidAngle = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * rand[particleId] + s_dt * s_dt * pTorque[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateRigidAngle);
  // update particle velocity with overdamped active dynamics
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  const double *pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
  kernelUpdateActiveParticleVel<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pVel, pForce, pAngle, config.driving, mobility);
}
