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
// wall updates
__global__ void kernelUpdateWallPos(double* wPos, const double* wVel, const double timeStep);
__global__ void kernelUpdateWallVel(double* wVel, const double* wForce, const double timeStep);


//************************* soft particle langevin ***************************//
void SoftParticleLangevin::integrate() {
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateThermalVel();
  updateVelocity(0.5 * sp_->dt);
  sp_->checkReflectiveWall();
}

void SoftParticleLangevin::injectKineticEnergy() {
  double amplitude(sqrt(config.Tinject));
  // generate random numbers between 0 and noise for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + sp_->numParticles * sp_->nDim, sp_->d_particleVel.begin(), gaussNum(0.f,amplitude));
}

void SoftParticleLangevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(sp_->nDim);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  double *pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinAddThermostatForces = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pForce[particleId * s_nDim + dim] += s_noise * rand[particleId * s_nDim + dim] - s_gamma * pVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinAddThermostatForces);
}

void SoftParticleLangevin::updateVelocity(double timeStep) {
  double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
  kernelUpdateParticleVel<<<sp_->dimGrid, sp_->dimBlock>>>(pVel, pForce, timeStep);

  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::mobile || sp_->simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    double* wVel = thrust::raw_pointer_cast(&(sp_->d_wallVel[0]));
    const double* wForce = thrust::raw_pointer_cast(&(sp_->d_wallForce[0]));
    kernelUpdateWallVel<<<sp_->dimGrid, sp_->dimBlock>>>(wVel, wForce, timeStep);
  }

  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    sp_->wallOmega += (sp_->wallAlpha - gamma * sp_->wallOmega) * timeStep;
  }
}

void SoftParticleLangevin::updatePosition(double timeStep) {
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  kernelUpdateParticlePos<<<sp_->dimGrid, sp_->dimBlock>>>(pPos, pVel, timeStep);

  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::mobile || sp_->simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    double* wPos = thrust::raw_pointer_cast(&(sp_->d_wallPos[0]));
    const double* wVel = thrust::raw_pointer_cast(&(sp_->d_wallVel[0]));
    kernelUpdateWallPos<<<sp_->dimGrid, sp_->dimBlock>>>(wPos, wVel, timeStep);
    
    if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
      double s_dt(timeStep);
      double s_el(sp_->el);
      double s_lgamma(sp_->lgamma);
      auto r = thrust::counting_iterator<long>(0);
      const double *wLength = thrust::raw_pointer_cast(&(sp_->d_wallLength[0]));
      double *rLength = thrust::raw_pointer_cast(&(sp_->d_restLength[0]));
      auto addPlasticRelaxation = [=] __device__ (long wallId) {
        rLength[wallId] -= s_dt * (rLength[wallId] - wLength[wallId]) * s_el / s_lgamma;
      };

      thrust::for_each(r, r + sp_->numWall, addPlasticRelaxation);
    }
  }
  
  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    sp_->wallAngle += sp_->wallOmega * timeStep;
    // apply rotation to wall monomers
    long s_nDim(sp_->nDim);
    double s_wallAngle(sp_->wallAngle);
    auto s = thrust::counting_iterator<long>(0);
    double* wPos = thrust::raw_pointer_cast(&(sp_->d_wallPos[0]));
    auto rotateWallMonomers = [=] __device__ (long wallId) {
      double newPos[MAXDIM];
			newPos[0] = wPos[wallId * s_nDim] * cos(s_wallAngle) - wPos[wallId * s_nDim + 1] * sin(s_wallAngle);
			newPos[1] = wPos[wallId * s_nDim] * sin(s_wallAngle) + wPos[wallId * s_nDim + 1] * cos(s_wallAngle);
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        wPos[wallId * s_nDim + dim] = newPos[dim];
      }
    };

    thrust::for_each(s, s + sp_->numWall, rotateWallMonomers);
  }
}

void SoftParticleLangevin::conserveMomentum() {
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> particleVel_x(sp_->d_particleVel.begin(), sp_->d_particleVel.end(), 2);
  strided_range<Iterator> particleVel_y(sp_->d_particleVel.begin() + 1, sp_->d_particleVel.end(), 2);
  double meanVelx = thrust::reduce(particleVel_x.begin(), particleVel_x.end(), double(0), thrust::plus<double>()) / sp_->numParticles;
  double meanVely = thrust::reduce(particleVel_y.begin(), particleVel_y.end(), double(0), thrust::plus<double>()) / sp_->numParticles;

  long s_nDim(sp_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  auto subtractParticleDrift = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[particleId * s_nDim + dim] -= ((1 - dim) * meanVelx + dim * meanVely);
    }
  };

  thrust::for_each(r, r + sp_->numParticles, subtractParticleDrift);
}

//************************* soft particle langevin ***************************//
void SoftParticleLangevin2::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  sp_->checkReflectiveWall();
}

void SoftParticleLangevin2::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + sp_->numParticles * sp_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
}

void SoftParticleLangevin2::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_noise(noise);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - s_gamma * pVel[pId * s_nDim + dim] + s_noise * rand[pId * s_nDim + dim]) -
      s_dt * s_dt * s_gamma * 0.5 * (pForce[pId * s_nDim + dim] - s_gamma * pVel[pId * s_nDim + dim]) -
      s_dt * s_dt * s_gamma * s_noise * (0.5 * rand[pId * s_nDim + dim] + rando[pId * s_nDim + dim] / sqrt(3));
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateParticleVel);
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
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + 0.5 * s_dt * s_dt * rando[pId * s_nDim + dim] / sqrt(3);
    }
  };

  thrust::for_each(r, r + sp_->numParticles, langevinUpdateParticlePos);
}

//************** soft particle langevin with massive particles ***************//
void SoftParticleLangevinSubset::integrate() {
  updateThermalVel();
  updateVelocity(0.5*sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5*sp_->dt);
  sp_->checkReflectiveWall();
}

void SoftParticleLangevinSubset::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + sp_->numParticles * sp_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
}

void SoftParticleLangevinSubset::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - s_gamma * pVel[pId * s_nDim + dim] + s_noise * rand[pId * s_nDim + dim]) -
      s_dt * s_dt * s_gamma * 0.5 * (pForce[pId * s_nDim + dim] - s_gamma * pVel[pId * s_nDim + dim]) -
      s_dt * s_dt * s_gamma * s_noise * (0.5 * rand[pId * s_nDim + dim] + rando[pId * s_nDim + dim] / sqrt(3));
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

void SoftParticleLangevinSubset::updatePosition(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	const double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto langevinUpdateParticlePos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + 0.5 * s_dt * s_dt * rando[pId * s_nDim + dim] / sqrt(3);
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

void SoftParticleLangevinSubset::conserveMomentum() {
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> particleVel_x(sp_->d_particleVel.begin() + firstIndex * sp_->nDim, sp_->d_particleVel.end(), 2);
  strided_range<Iterator> particleVel_y(sp_->d_particleVel.begin() + 1 + firstIndex * sp_->nDim, sp_->d_particleVel.end(), 2);
  double meanVelx = thrust::reduce(particleVel_x.begin(), particleVel_x.end(), double(0), thrust::plus<double>()) / (sp_->numParticles - firstIndex);
  double meanVely = thrust::reduce(particleVel_y.begin(), particleVel_y.end(), double(0), thrust::plus<double>()) / (sp_->numParticles - firstIndex);

  long s_nDim(sp_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
  auto subtractParticleDrift = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[particleId * s_nDim + dim] -= ((1 - dim) * meanVelx + dim * meanVely);
    }
  };

  thrust::for_each(r + firstIndex, r + firstIndex + sp_->numParticles, subtractParticleDrift);
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
  sp_->checkReflectiveWall();
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
  sp_->checkReflectiveWall();
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
}

void SoftParticleLangevinFlow::updateVelocity(double timeStep) {
  long s_nDim(sp_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  const double *flowVel = thrust::raw_pointer_cast(&(sp_->d_flowVel[0]));
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));

  auto langevinUpdateParticleFlowVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] + s_gamma * (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]) + s_noise * rand[pId * s_nDim + dim]) -
      s_dt * s_dt * s_gamma * 0.5 * (pForce[pId * s_nDim + dim] + s_gamma * (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim])) -
      s_dt * s_dt * s_gamma * s_noise * (0.5 * rand[pId * s_nDim + dim] + rando[pId * s_nDim + dim] / sqrt(3));
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
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] + s_gamma * (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]));
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * s_gamma * (pForce[pId * s_nDim + dim] + s_gamma * (flowVel[pId * s_nDim + dim] - pVel[pId * s_nDim + dim]));
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
  sp_->checkReflectiveWall();
}

//**************** soft particle nve with velocity rescaling *****************//
void SoftParticleNVERescale::integrate() {
  injectKineticEnergy();
  updateVelocity(0.5 * sp_->dt);
  updatePosition(sp_->dt);
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateVelocity(0.5 * sp_->dt);
  sp_->checkReflectiveWall();
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
  sp_->checkReflectiveWall();
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
  sp_->checkReflectiveWall();
}

void SoftParticleNoseHoover::updateVelocity(double timeStep) {
  // update nose hoover damping
  gamma += (timeStep / (2 * mass)) * (sp_->getParticleKineticEnergy() - (sp_->nDim * sp_->numParticles + 1) * config.Tinject / 2);
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
  sp_->checkReflectiveWall();
}

void SoftParticleDoubleNoseHoover::injectKineticEnergy() {
  double amplitude1(sqrt(config.Tinject));
  double amplitude2(sqrt(config.driving));
  // generate random numbers between 0 and noise for thermal noise
  thrust::counting_iterator<long> index_sequence_begin1(lrand48());
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + sp_->num1 * sp_->nDim, sp_->d_particleVel.begin(), gaussNum(0.f,amplitude1));
  thrust::counting_iterator<long> index_sequence_begin2(lrand48());
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + (sp_->numParticles - sp_->num1) * sp_->nDim, sp_->d_particleVel.begin() + sp_->num1 * sp_->nDim, gaussNum(0.f,amplitude2));
}

void SoftParticleDoubleNoseHoover::updateVelocity(double timeStep) {
  // update nose hoover damping
  std::tuple<double, double, double> ekins = sp_->getParticleKineticEnergy12();
  gamma1 += (timeStep / (2 * mass)) * (get<0>(ekins) - (sp_->nDim * sp_->num1 + 1) * config.Tinject / 2);//T1
  gamma2 += (timeStep / (2 * mass)) * (get<1>(ekins) - (sp_->nDim * (sp_->numParticles - sp_->num1) + 1) * config.driving / 2);//T2
  double s_gamma1(gamma1);
  double s_gamma2(gamma2);
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
  gamma1 += (sp_->dt / (2 * mass)) * (get<0>(ekins) - (sp_->nDim * sp_->num1 + 1) * config.Tinject / 2);//T1
  gamma2 += (sp_->dt / (2 * mass)) * (get<1>(ekins) - (sp_->nDim * (sp_->numParticles - sp_->num1) + 1) * config.driving / 2);//T2
  double s_gamma1(gamma1);
  double s_gamma2(gamma2);
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

//**************************** brownian integrator *****************************//
void SoftParticleBrownian::integrate() {
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateThermalVel();
  sp_->checkReflectiveWall();
  updatePosition(sp_->dt);
}

void SoftParticleBrownian::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + sp_->numParticles * sp_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  // assign overdamped velocity
  long s_nDim(sp_->nDim);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto updateBrownianVel = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
		  pVel[particleId * s_nDim + dim] = (pForce[particleId * s_nDim + dim] + s_noise * rand[particleId * s_nDim + dim]) / s_gamma;
    }
  };

  thrust::for_each(r, r + sp_->numParticles, updateBrownianVel);

  // wall velocity update
  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::mobile || sp_->simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    auto s = thrust::counting_iterator<long>(0);
    double* wVel = thrust::raw_pointer_cast(&(sp_->d_wallVel[0]));
    const double* wForce = thrust::raw_pointer_cast(&(sp_->d_wallForce[0]));

    auto updateWallVelocity =  [=] __device__ (long wallId) {
      wVel[wallId] = wForce[wallId] / s_gamma;
    };

    thrust::for_each(s, s + sp_->numWall, updateWallVelocity);
  }

  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    sp_->wallOmega = sp_->wallAlpha / gamma;
  }
}

//**************************** driven brownian integrator *****************************//
void SoftParticleDrivenBrownian::integrate() {
  sp_->checkParticleNeighbors();
  sp_->calcParticleForceEnergy();
  updateThermalVel();
  sp_->checkReflectiveWall();
  updatePosition(sp_->dt);
}

void SoftParticleDrivenBrownian::updateThermalVel() {
  // assign overdamped velocity as total force over damping
  long s_nDim(sp_->nDim);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
  double *pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));

  auto updateActiveBrownianVel = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      // active noise has already been added to the force
		  pVel[particleId * s_nDim + dim] = pForce[particleId * s_nDim + dim] / s_gamma;
    }
  };

  thrust::for_each(r, r + sp_->numParticles, updateActiveBrownianVel);

  // wall velocity update
  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::mobile || sp_->simControl.boundaryType == simControlStruct::boundaryEnum::plastic) {
    auto s = thrust::counting_iterator<long>(0);
    double* wVel = thrust::raw_pointer_cast(&(sp_->d_wallVel[0]));
    const double* wForce = thrust::raw_pointer_cast(&(sp_->d_wallForce[0]));

    auto updateWallVelocity =  [=] __device__ (long wallId) {
      wVel[wallId] = wForce[wallId] / s_gamma;
    };

    thrust::for_each(s, s + sp_->numWall, updateWallVelocity);
  }

  if(sp_->simControl.boundaryType == simControlStruct::boundaryEnum::rigid) {
    sp_->wallOmega = sp_->wallAlpha / gamma;
  }
}