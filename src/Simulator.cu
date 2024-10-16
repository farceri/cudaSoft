//
// Author: Francesco Arceri
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
// position updates
__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep);
__global__ void kernelUpdateParticleVel(double* pVel, const double* pForce, const double timeStep);
// angle updates
__global__ void kernelUpdateParticleAngle(double* pAngle, const double* pOmega, const double timeStep);
__global__ void kernelUpdateParticleOmega(double* pOmega, const double* pAlpha, const double timeStep);
// momentum conservation
__global__ void kernelSumParticleVelocity(double* pVel, double* velSum);
__global__ void kernelSubtractParticleDrift(double* pVel, double* velSum);
__global__ void kernelSubsetSumParticleVelocity(double* pVel, double* velSum, long firstId);
__global__ void kernelSubsetSubtractParticleDrift(double* pVel, double* velSum, long firstId);


//************************* soft particle langevin ***************************//
void SoftParticleLangevin::integrate() {
  updateVelocity(0.5*sp_->dt);
  updatePosition(0.5*sp_->dt);
  updateThermalVel();
  updatePosition(0.5*sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5 * sp_->dt);
  conserveMomentum();
}

void SoftParticleLangevin::injectKineticEnergy() {
  double amplitude(sqrt(config.Tinject));
  // generate random numbers between 0 and noiseVar for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + sp_->numParticles * sp_->nDim, sp_->d_particleVel.begin(), gaussNum(0.f,amplitude));
  conserveMomentum();
}

void SoftParticleLangevin::updatePosition(double timeStep) {
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  kernelUpdateParticlePos<<<sp_->dimGrid, sp_->dimBlock>>>(pPos, pVel, timeStep);
}

void SoftParticleLangevin::updateVelocity(double timeStep) {
  double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
  kernelUpdateParticleVel<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, pForce, timeStep);
}

void SoftParticleLangevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + sp_->numParticles * sp_->nDim, d_thermalVel.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(sp_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalVel = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[particleId * s_nDim + dim] = s_lcoeff1 * pVel[particleId * s_nDim + dim] + s_lcoeff2 * thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateThermalVel);
}

void SoftParticleLangevin::conserveMomentum() {
  d_velSum.resize(sp_->nDim);
  thrust::fill(d_velSum.begin(), d_velSum.end(), double(0));
  double *velSum = thrust::raw_pointer_cast(&d_velSum[0]);
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  kernelSumParticleVelocity<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, velSum);
  kernelSubtractParticleDrift<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, velSum);
}

//************************* soft particle langevin ***************************//
void SoftParticleLangevin2::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

void SoftParticleLangevin2::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + sp_->numParticles * sp_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(sp_->nDim);
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

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateThermalNoise);
}

void SoftParticleLangevin2::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) + rand[pId * s_nDim + dim] -
      0.5 * s_dt * s_dt * s_gamma * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateParticleVel);
  if(sp_->simControl.particleType == simControlStruct::particleEnum::vicsek) {
    double* pOmega = thrust::raw_pointer_cast(&(sp_->d_particleOmega[0]));
    const double* pAlpha = thrust::raw_pointer_cast(&(sp_->d_particleAlpha[0]));
    kernelUpdateParticleOmega<<<sp_->dimGrid, sp_->dimBlock>>>(pOmega, pAlpha, timeStep);
  }
}

void SoftParticleLangevin2::updatePosition(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto langevinUpdateParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateParticlePos);
  if(sp_->simControl.particleType == simControlStruct::particleEnum::vicsek) {
    double* pAngle = thrust::raw_pointer_cast(&(sp_->d_particleAngle[0]));
    const double* pOmega = thrust::raw_pointer_cast(&(sp_->d_particleOmega[0]));
    kernelUpdateParticleAngle<<<sp_->dimGrid, sp_->dimBlock>>>(pAngle, pOmega, timeStep);
  }
}

//************** soft particle langevin with massive particles ***************//
void SoftParticleLangevinSubSet::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

void SoftParticleLangevinSubSet::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + sp_->numParticles * sp_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(sp_->nDim);
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

  thrust::for_each(r + firstIndex, r + sp_->numParticles, langevinUpdateThermalNoise);
}

void SoftParticleLangevinSubSet::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) * s_gamma;
      pVel[pId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r + firstIndex, r + sp_->numParticles, langevinUpdateParticleVel);
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

void SoftParticleLangevinSubSet::updatePosition(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto langevinUpdateParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r + firstIndex, r + sp_->numParticles, langevinUpdateParticlePos);
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

void SoftParticleLangevinSubSet::conserveMomentum() {
  d_velSum.resize(sp_->nDim);
  thrust::fill(d_velSum.begin(), d_velSum.end(), double(0));
  double *velSum = thrust::raw_pointer_cast(&d_velSum[0]);
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  kernelSubsetSumParticleVelocity<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, velSum, firstIndex);
  kernelSubsetSubtractParticleDrift<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, velSum, firstIndex);
}

//*************** soft particle langevin with external field *****************//
void SoftParticleLangevinExtField::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  sp_->addExternalParticleForce();
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

//******* soft particle langevin with perturbation on first particles ********//
void SoftParticleLangevinPerturb::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  sp_->addConstantParticleForce(extForce, firstIndex);
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

//****************** soft particle langevin with fluid flow ******************//
void SoftParticleLangevinFlow::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->calcFlowVelocity();
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

void SoftParticleLangevinFlow::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  const double *flowVel = thrust::raw_pointer_cast(&(sp_->d_flowVel[0]));
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleFlowVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] + (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]) * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] + (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]) * s_gamma) * s_gamma;
      pVel[pId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateParticleFlowVel);
  //kernelConserveParticleMomentum<<<1, sp_->dimBlock>>>(pVel);
}

//*************** soft particle damped dynamics with fluid flow ***************//
void SoftParticleFlow::integrate() {
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->calcFlowVelocity();
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  //conserveMomentum();
}

void SoftParticleFlow::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *flowVel = thrust::raw_pointer_cast(&(sp_->d_flowVel[0]));
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto updateParticleFlowVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] + (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]) * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] + (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]) * s_gamma) * s_gamma;
    }
  };

  thrust::for_each(r, r + sp_->numParticles, updateParticleFlowVel);
  //kernelConserveParticleMomentum<<<1, sp_->dimBlock>>>(pVel);
}

void SoftParticleFlow::updatePosition(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto updateParticleFlowPos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, updateParticleFlowPos);
}

//**************************** soft particle nve *****************************//
void SoftParticleNVE::integrate() {
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5 * sp_->dt);
  //conserveMomentum();
}

//**************** soft particle nve with velocity rescaling *****************//
void SoftParticleNVERescale::integrate() {
  injectKineticEnergy();
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5 * sp_->dt);
  //conserveMomentum();
}

void SoftParticleNVERescale::injectKineticEnergy() {
  double scale = sqrt(config.Tinject / sp_->getParticleTemperature());
  long s_nDim(sp_->nDim);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto scaleParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] *= scale;
    }
  };

  thrust::for_each(r, r + sp_->numParticles, scaleParticleVel);
}

//**************** soft particle nve with velocity rescaling *****************//
void SoftParticleNVEDoubleRescale::integrate() {
  injectKineticEnergy();
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5 * sp_->dt);
  //conserveMomentum();
}

void SoftParticleNVEDoubleRescale::injectKineticEnergy() {
  std::tuple<double, double, double> Temps = sp_->getParticleT1T2();
  double scale1 = sqrt(config.Tinject / get<0>(Temps));
  double scale2 = sqrt(config.driving / get<1>(Temps));
  long s_nDim(sp_->nDim);
  long s_num1(sp_->num1);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto doubleScaleParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      if(pId < s_num1) {
        pVel[pId * s_nDim + dim] *= scale1;
      } else {
        pVel[pId * s_nDim + dim] *= scale2;
      }
    }
  };

  thrust::for_each(r, r + sp_->num1, doubleScaleParticleVel);
}

//************************ soft particle Nose Hoover **************************//
void SoftParticleNoseHoover::integrate() {
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateThermalVel();
}

void SoftParticleNoseHoover::updateVelocity(double timeStep) {
  // update nose hoover damping
  gamma += (sp_->dt / (2 * mass)) * (sp_->getParticleKineticEnergy() - (sp_->nDim * sp_->numParticles + 1) * config.Tinject / 2);
  double s_gamma(gamma);
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto noseHooverUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma);
    }
  };

  thrust::for_each(r, r + sp_->numParticles, noseHooverUpdateParticleVel);
}

void SoftParticleNoseHoover::updateThermalVel() {
  // update nose hoover damping
  gamma += (sp_->dt / (2 * mass)) * (sp_->getParticleKineticEnergy() - (sp_->nDim * sp_->numParticles + 1) * config.Tinject / 2);
  double s_gamma(gamma);
  long s_nDim(sp_->nDim);
  double s_dt(sp_->dt);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto noseHooverSecondUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] = (pVel[pId * s_nDim + dim] + 0.5 * s_dt * pForce[pId * s_nDim + dim]) / (1 + 0.5 * s_dt * s_gamma);
    }
  };

  thrust::for_each(r, r + sp_->numParticles, noseHooverSecondUpdateParticleVel);
}

//******************** soft particle double T Nose Hoover *******************//
void SoftParticleDoubleNoseHoover::integrate() {
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateThermalVel();
}

void SoftParticleDoubleNoseHoover::injectKineticEnergy() {
  double amplitude1(sqrt(config.Tinject));
  double amplitude2(sqrt(config.driving));
  // generate random numbers between 0 and noiseVar for thermal noise
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->num1 * sp_->nDim, sp_->d_particleVel.begin(), gaussNum(0.f,amplitude1));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + (sp_->numParticles - sp_->num1) * sp_->nDim, sp_->d_particleVel.begin() + sp_->num1 * sp_->nDim, gaussNum(0.f,amplitude2));
  conserveMomentum();
}

void SoftParticleDoubleNoseHoover::updateVelocity(double timeStep) {
  // update nose hoover damping
  std::tuple<double, double, double> ekins = sp_->getParticleKineticEnergy12();
  lcoeff1 += (sp_->dt / (2 * mass)) * (get<0>(ekins) - (sp_->nDim * sp_->num1 + 1) * config.Tinject / 2);//T1
  lcoeff2 += (sp_->dt / (2 * mass)) * (get<1>(ekins) - (sp_->nDim * (sp_->numParticles - sp_->num1) + 1) * config.driving / 2);//T2
  double s_gamma1(lcoeff1);
  double s_gamma2(lcoeff2);
  long s_nDim(sp_->nDim);
  long s_num1(sp_->num1);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto doubleNoseHooverUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      if(pId < s_num1) {
        pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma1);
      } else {
        pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma2);
      }
    }
  };

  thrust::for_each(r, r + sp_->numParticles, doubleNoseHooverUpdateParticleVel);
}

void SoftParticleDoubleNoseHoover::updateThermalVel() {
  // update nose hoover damping
  std::tuple<double, double, double> ekins = sp_->getParticleKineticEnergy12();
  lcoeff1 += (sp_->dt / (2 * mass)) * (get<0>(ekins) - (sp_->nDim * sp_->num1 + 1) * config.Tinject / 2);//T1
  lcoeff2 += (sp_->dt / (2 * mass)) * (get<1>(ekins) - (sp_->nDim * (sp_->numParticles - sp_->num1) + 1) * config.driving / 2);//T2
  double s_gamma1(lcoeff1);
  double s_gamma2(lcoeff2);
  long s_nDim(sp_->nDim);
  long s_num1(sp_->num1);
  double s_dt(sp_->dt);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto doubleNoseHooverSecondUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      if(pId < s_num1) {
        pVel[pId * s_nDim + dim] = (pVel[pId * s_nDim + dim] + 0.5 * s_dt * pForce[pId * s_nDim + dim]) / (1 + 0.5 * s_dt * s_gamma1);
      } else {
        pVel[pId * s_nDim + dim] = (pVel[pId * s_nDim + dim] + 0.5 * s_dt * pForce[pId * s_nDim + dim]) / (1 + 0.5 * s_dt * s_gamma2);
      }
    }
  };

  thrust::for_each(r, r + sp_->numParticles, doubleNoseHooverSecondUpdateParticleVel);
}

