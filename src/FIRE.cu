//
// Author: Francesco Arceri
// Date:   10-26-2021
//
// FUNCTIONS FOR FIRE CLASS

#include "../include/FIRE.h"
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>

using namespace std;

//********************** constructor and deconstructor ***********************//
FIRE::FIRE(SP2D * spPtr):sp_(spPtr){
	// Note that mass is used only for particle-level FIRE
	d_mass.resize(sp_->numParticles * sp_->nDim);
	// Set variables to zero
	thrust::fill(d_mass.begin(), d_mass.end(), double(0));
}

FIRE::~FIRE() {
	d_mass.clear();
	d_velSquared.clear();
	d_forceSquared.clear();
};

// initilize the minimizer
void FIRE::initMinimizer(double a_start_, double f_dec_, double f_inc_, double f_a_, double fire_dt_, double fire_dt_max_, double a_, long minStep_, long numStep_, long numDOF_) {
	a_start = a_start_;
	f_dec = f_dec_;
	f_inc = f_inc_;
	f_a = f_a_;
	fire_dt = fire_dt_;
	fire_dt_max = fire_dt_max_;
	a = a_;
	minStep = minStep_;
	numStep = numStep_;
	d_velSquared.resize(numDOF_ * sp_->nDim);
	d_forceSquared.resize(numDOF_ * sp_->nDim);
}

//*************************** particle minimizer *****************************//
// update position and velocity in response to an applied force
void FIRE::updateParticlePositionAndVelocity() {
	long f_nDim(sp_->nDim);
	double d_fire_dt(fire_dt);
	auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&(sp_->d_particlePos[0]));
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
	const double* mass = thrust::raw_pointer_cast(&d_mass[0]);

	auto perParticleUpdatePosAndVel = [=] __device__ (long particleId) {
		double totalForce(0);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pVel[particleId * f_nDim + dim] += 0.5 * d_fire_dt * pForce[particleId * f_nDim + dim] / mass[particleId * f_nDim + dim];
			pPos[particleId * f_nDim + dim] += d_fire_dt * pVel[particleId * f_nDim + dim];
			totalForce += pForce[particleId * f_nDim + dim];
		}
		//If the total force on a particle is zero, then zero out the velocity as well
		if (totalForce == 0) {
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < f_nDim; dim++) {
				pVel[particleId * f_nDim + dim] = 0;
			}
		}
	};

	thrust::for_each(r, r + sp_->numParticles, perParticleUpdatePosAndVel);
}

// update velocity in response to an applied force and return the maximum displacement in the previous step
void FIRE::updateParticleVelocity() {
	long f_nDim(sp_->nDim);
	double d_fire_dt(fire_dt);
	auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
	const double* mass = thrust::raw_pointer_cast(&d_mass[0]);

	auto perParticleUpdateVel = [=] __device__ (long particleId) {
		double totalForce(0);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pVel[particleId * f_nDim + dim] += 0.5 * d_fire_dt * pForce[particleId * f_nDim + dim] / mass[particleId * f_nDim + dim];
			totalForce += pForce[particleId * f_nDim + dim];
		}
		//If the total force on a particle is zero, then zero out the velocity as well
		if (totalForce == 0) {
			for (long dim = 0; dim < f_nDim; dim++) {
				pVel[particleId * f_nDim + dim] = 0;
			}
		}
	};

	thrust::for_each(r, r + sp_->numParticles, perParticleUpdateVel);
}

// bend the velocity towards the force
void FIRE::bendParticleVelocityTowardsForce() {
	double velNormSquared = 0, forceNormSquared = 0;
	// get the dot product between the velocity and the force
	double vDotF = double(thrust::inner_product(sp_->d_particleVel.begin(), sp_->d_particleVel.end(), sp_->d_particleForce.begin(), double(0)));
	//cout << "FIRE::bendVelocityTowardsForceFIRE: vDotF = " << setprecision(precision) << vDotF << endl;
	if (vDotF < 0) {
		// if vDotF is negative, then we are going uphill, so let's stop and reset
		thrust::fill(sp_->d_particleVel.begin(), sp_->d_particleVel.end(), double(0));
		numStep = 0;
		fire_dt = std::max(fire_dt * f_dec, fire_dt_max / 10); // go to a shorter dt
		a = a_start; // start fresh with a more radical mixing between force and velocity
	} else if (numStep > minStep) {
		// if enough time has passed then let's start to increase the inertia
		fire_dt = std::min(fire_dt * f_inc, fire_dt_max);
		a *= f_a; // increase the inertia
	}
	// calculate the ratio of the norm squared of the velocity and the force
  thrust::transform(sp_->d_particleVel.begin(), sp_->d_particleVel.end(), d_velSquared.begin(), square());
  thrust::transform(sp_->d_particleForce.begin(), sp_->d_particleForce.end(), d_forceSquared.begin(), square());
	velNormSquared = thrust::reduce(d_velSquared.begin(), d_velSquared.end(), double(0), thrust::plus<double>());
	forceNormSquared = thrust::reduce(d_forceSquared.begin(), d_forceSquared.end(), double(0), thrust::plus<double>());
	// check FIRE convergence
	if (forceNormSquared == 0) {
		// if the forceNormSq is zero, then there is no force and we are done, so zero out the velocity
		cout << "FIRE::bendVelocityTowardsForceFIRE: forceNormSquared is zero" << endl;
		thrust::fill(sp_->d_particleVel.begin(), sp_->d_particleVel.end(), double(0));
	} else {
		double velForceNormRatio = sqrt(velNormSquared / forceNormSquared);
		double f_a(a);
		auto r = thrust::counting_iterator<long>(0);
		double* pVel = thrust::raw_pointer_cast(&(sp_->d_particleVel[0]));
		const double* pForce = thrust::raw_pointer_cast(&(sp_->d_particleForce[0]));
		auto perDOFBendParticleVelocity = [=] __device__ (long i) {
			pVel[i] = (1 - f_a) * pVel[i] + f_a * pForce[i] * velForceNormRatio;
		};

		thrust::for_each(r, r + sp_->d_particleVel.size(), perDOFBendParticleVelocity);
	}
}

// set the mass for each degree of freedom
void FIRE::setParticleMass() {
	d_mass.resize(sp_->numParticles * sp_->nDim);
	for (long particleId = 0; particleId < sp_->numParticles; particleId++) {
		for (long dim = 0; dim < sp_->nDim; dim++) {
			d_mass[particleId * sp_->nDim + dim] = PI / (sp_->d_particleRad[particleId] * sp_->d_particleRad[particleId]);
		}
	}
}

// set FIRE time step
void FIRE::setFIRETimeStep(double fire_dt_) {
	fire_dt = fire_dt_;
	fire_dt_max = 10 * fire_dt_;
}

// Run the inner loop of the FIRE algorithm
void FIRE::minimizerParticleLoop() {
	// Move the system forward, based on the previous velocities and forces
	updateParticlePositionAndVelocity();
	// Calculate the new set of forces at the new step
	sp_->calcParticleForceEnergy();
	// update the velocity based on the current forces
	updateParticleVelocity();
	// Bend the velocity towards the force
	bendParticleVelocityTowardsForce();
	// Increase the number of steps since the last restart
	numStep++;
}
