//
// Author: Francesco Arceri
// Date:   10-26-2021
//
// HEADER FILE FOR FIRE CLASS

#ifndef FIRE_H_
#define FIRE_H_

#include "SP2D.h"
#include "defs.h"
#include <thrust/device_vector.h>

class SP2D;

class FIRE
{
public:
	SP2D * sp_;  //Pointer to the enclosing class

	FIRE() = default;
	FIRE(SP2D * spPtr);
	~FIRE();
	// Global FIRE params
	double a_start;
	double a;
	double f_dec;
	double f_inc;
	double f_a;
	double fire_dt_max;
	double fire_dt;
	double cutDistance;
	long minStep;
	long numStep;
	// particle variables
	thrust::device_vector<double> d_mass;
	// fire variables
	thrust::device_vector<double> d_velSquared;
	thrust::device_vector<double> d_forceSquared;

	// initialize minimizer for particles
	void initMinimizer(double a_start_, double f_dec_, double f_inc_, double f_a_, double fire_dt_, double dt_max_, double a_, long minStep_, long numStep_, long numDOF_);

	//******************** functions for particle minimizer ********************//
	// update position and velocity in response to an applied force
	void updateParticlePositionAndVelocity();
	// update velocity in response to an applied force and return the maximum displacement in the previous step
	void updateParticleVelocity();
	// bend the velocity towards the force
	void bendParticleVelocityTowardsForce();
	// set the mass for each degree of freedom
	void setParticleMass();
	// set FIRE time step
	void setFIRETimeStep(double timeStep);
	// fire minimizer loop for particles
	void minimizerParticleLoop();

};

#endif /* FIRE_H_ */
