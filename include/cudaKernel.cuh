//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// KERNEL FUNCTIONS THAT ACT ON THE DEVICE(GPU)

#ifndef PACKINGDKERNEL_CUH_
#define PACKINGKERNEL_CUH_

#include "defs.h"
#include <stdio.h>

__constant__ simControlStruct d_simControl;

__constant__ long d_dimBlock;
__constant__ long d_dimGrid;

__constant__ double* d_boxSizePtr;

//leesEdwards shift
__constant__ double d_LEshift;

__constant__ long d_nDim;
__constant__ long d_numParticles;

// time step
__constant__ double d_dt;
// dimensionality factor
__constant__ double d_rho0;
// energy costs
__constant__ double d_ec; // interaction
__constant__ double d_l1; // depth
__constant__ double d_l2; // range
// Lennard-Jones constants
__constant__ double d_LJcutoff;
__constant__ double d_LJecut;
// Gravity
__constant__ double d_gravity;
__constant__ double d_ew; // wall

// particle neighborList
__constant__ long* d_partNeighborListPtr;
__constant__ long* d_partMaxNeighborListPtr;
__constant__ long d_partNeighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_partMaxNeighbors;


inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	double delta = x1 - x2, size = d_boxSizePtr[dim];
	return delta - size * round(delta / size); //round for distance, floor for position
}

//for leesEdwards need to handle first two dimensions together
inline __device__ double pbcDistanceLE(const double x1, const double y1, const double x2, const double y2) {
	double deltax = (x1 - x2);
	double rounded = rint(deltax); //need to store for lees-edwards BC
	deltax -= rounded;
	double deltay = (y1 - y2);
	deltay = deltay - rounded*d_LEshift - rint(deltay - rounded*d_LEshift);
	return deltay;
}

inline __device__ double calcNorm(const double* segment) {
  double normSq = 0.;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return sqrt(normSq);
}

inline __device__ double calcNormSq(const double* segment) {
  double normSq = 0.;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return normSq;
}

inline __device__ double calcDistance(const double* thisVec, const double* otherVec) {
  double delta, distanceSq = 0., deltay, deltax, shifty;
	switch (d_simControl.geometryType) {
		case simControlStruct::geometryEnum::normal:
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			distanceSq += delta * delta;
		}
		break;
		case simControlStruct::geometryEnum::leesEdwards:
		deltay = thisVec[1] - otherVec[1];
		shifty = round(deltay / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltax = thisVec[0] - otherVec[0];
		deltax -= shifty * d_LEshift;
		deltax -= round(deltax / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltay -= shifty;
		distanceSq = deltax * deltax + deltay * deltay;
		break;
		case simControlStruct::geometryEnum::fixedBox:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			distanceSq += delta * delta;
		}
		break;
		case simControlStruct::geometryEnum::fixedSides:
		deltay = thisVec[1] - otherVec[1];
		deltax = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq = deltax * deltax + deltay * deltay;
		break;
	}
  return sqrt(distanceSq);
}

inline __device__ double calcDeltaAndDistance(const double* thisVec, const double* otherVec, double* deltaVec) {
	double delta, distanceSq = 0., shifty;
	switch (d_simControl.geometryType) {
		case simControlStruct::geometryEnum::normal:
		#pragma unroll (MAXDIM)
	  for (long dim = 0; dim < d_nDim; dim++) {
	    delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			deltaVec[dim] = delta;
	    distanceSq += delta * delta;
	  }
		break;
		case simControlStruct::geometryEnum::leesEdwards:
		deltaVec[1] = thisVec[1] - otherVec[1];
		shifty = round(deltaVec[1] / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltaVec[0] = thisVec[0] - otherVec[0];
		deltaVec[0] -= shifty * d_LEshift;
		deltaVec[0] -= round(deltaVec[0] / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltaVec[1] -= shifty;
		distanceSq = deltaVec[0] * deltaVec[0] + deltaVec[1] * deltaVec[1];
		break;
		case simControlStruct::geometryEnum::fixedBox:
		#pragma unroll (MAXDIM)
	  for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			deltaVec[dim] = delta;
			distanceSq += delta * delta;
		}
		break;
		case simControlStruct::geometryEnum::fixedSides:
		deltaVec[1] = thisVec[1] - otherVec[1];
		deltaVec[0] = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq = deltaVec[0] * deltaVec[0] + deltaVec[1] * deltaVec[1];
		break;
	}
  return sqrt(distanceSq);
}

inline __device__ double calcFixedBoundaryDistance(const double* thisVec, const double* otherVec) {
  double distanceSq = 0.;
  double delta;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    delta = thisVec[dim] - otherVec[dim];
    distanceSq += delta * delta;
  }
  return sqrt(distanceSq);
}

inline __device__ void getSegment(const double* thisVec, const double* otherVec, double* segment) {
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    segment[dim] = thisVec[dim] - otherVec[dim];
  }
}

inline __device__ void getParticlePos(const long pId, const double* pPos, double* tPos) {
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
		tPos[dim] = pPos[pId * d_nDim + dim];
	}
}

inline __device__ bool extractOtherParticle(const long particleId, const long otherId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	if ((particleId != otherId) && (otherId != -1)) {
		getParticlePos(otherId, pPos, otherPos);
		otherRad = pRad[particleId];
    return true;
  }
  return false;
}

inline __device__ bool extractOtherParticlePos(const long particleId, const long otherId, const double* pPos, double* otherPos) {
	if ((particleId != otherId) && (otherId != -1)) {
		getParticlePos(otherId, pPos, otherPos);
    return true;
  }
  return false;
}

inline __device__ bool extractParticleNeighbor(const long particleId, const long nListId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
  if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    for (long dim = 0; dim < d_nDim; dim++) {
      otherPos[dim] = pPos[otherId * d_nDim + dim];
    }
    otherRad = pRad[otherId];
    return true;
  }
  return false;
}

inline __device__ bool extractParticleNeighborPos(const long particleId, const long nListId, const double* pPos, double* otherPos) {
	long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
  if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    for (long dim = 0; dim < d_nDim; dim++) {
      otherPos[dim] = pPos[otherId * d_nDim + dim];
    }
    return true;
  }
  return false;
}

//***************************** force and energy *****************************//
inline __device__ double calcOverlap(const double* thisVec, const double* otherVec, const double radSum) {
	return (1 - calcDistance(thisVec, otherVec) / radSum);
}

inline __device__ double calcFixedBoundaryOverlap(const double* thisVec, const double* otherVec, const double radSum) {
  return (1 - calcFixedBoundaryDistance(thisVec, otherVec) / radSum);
}

inline __device__ void getNormalVector(const double* thisVec, double* normalVec) {
  normalVec[0] = thisVec[1];
  normalVec[1] = -thisVec[0];
}

inline __device__ double calcLJForceShift(const double radSum, const double radSum6) {
	double distance, distance6;
	distance = d_LJcutoff * radSum;
	distance6 = pow(distance, 6);
	return 24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
	//return 24 * d_ec * (2 * radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) / distance;
}

inline __device__ double calcGradMultiple(const double* thisPos, const double* otherPos, const double radSum) {
	double distance, overlap, distance6, radSum6, forceShift;
	distance = calcDistance(thisPos, otherPos);
	switch (d_simControl.potentialType) {
		case simControlStruct::potentialEnum::harmonic:
		overlap = 1 - distance / radSum;
		if(overlap > 0) {
			return d_ec * overlap / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::lennardJones:
		distance6 = pow(distance, 6);
		radSum6 = pow(radSum, 6);
		if (distance <= (d_LJcutoff * radSum)) {
			forceShift = calcLJForceShift(radSum, radSum6);
			return -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance + forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::WCA:
		distance6 = pow(distance, 6);
		radSum6 = pow(radSum, 6);
		if (distance <= (WCAcut * radSum)) {
			return -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::adhesive:
		overlap = 1 - distance / radSum;
		if (distance < (1 + d_l1) * radSum) {
			return d_ec * overlap / radSum;
		} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			return -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		} else {
			return 0;
		}
		break;
	}
}

inline __device__ double calcContactInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
  double distance, overlap, gradMultiple;
	//overlap = calcOverlap(thisPos, otherPos, radSum);
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	overlap = 1 - distance / radSum;
	if (overlap > 0) {
		gradMultiple = d_ec * overlap / radSum;
		#pragma unroll (MAXDIM)
	  for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	  return (0.5 * d_ec * overlap * overlap) * 0.5;
	}
	return 0.;
}

inline __device__ double calcWallContactInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
  double distance, overlap, gradMultiple;
	//overlap = calcOverlap(thisPos, otherPos, radSum);
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	overlap = 1 - distance / radSum;
	if (overlap > 0) {
		gradMultiple = d_ew * overlap / radSum;
		#pragma unroll (MAXDIM)
	  for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	  return (0.5 * d_ew * overlap * overlap);
	}
	return 0.;
}

inline __device__ double calcLJInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double distance, distance6, radSum6, forceShift, gradMultiple = 0, epot = 0.;
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	distance6 = pow(distance, 6);
	radSum6 = pow(radSum, 6);
	if (distance <= (d_LJcutoff * radSum)) {
		forceShift = calcLJForceShift(radSum, radSum6);
		gradMultiple = -24 * d_ec * radSum6 * (1 / distance6 - 2 * radSum6 / (distance6 * distance6)) / distance + forceShift;
		epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) - d_LJecut);
		epot -= 0.5 * abs(forceShift) * (distance - d_LJcutoff * radSum);
		//gradMultiple = 24 * d_ec * (2 * radSum12 / distance12 - radSum6 / distance6) / distance - forceShift;
		//epot = 0.5 * d_ec * (4 * (radSum12 / distance12 - radSum6 / distance6) - d_LJecut);
		//epot -= 0.5 * abs(forceShift) * (distance - d_LJcutoff * radSum);
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

inline __device__ double calcWCAInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double distance, distance6, radSum6, gradMultiple = 0, epot = 0;
	double delta[MAXDIM];
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	distance6 = pow(distance, 6);
	radSum6 = pow(radSum, 6);
	if (distance <= (WCAcut * radSum)) {
		gradMultiple = -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
		epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) + 1);
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

inline __device__ double calcAdhesiveInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double overlap, distance, gradMultiple = 0, epot = 0.;
	double delta[MAXDIM];
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	overlap = 1 - distance / radSum;
	if (distance < (1 + d_l1) * radSum) {
		gradMultiple = d_ec * overlap / radSum;
		epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
	} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
		gradMultiple = -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2)) * 0.5;
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

// particle-particle interaction
__global__ void kernelCalcParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    double thisRad, otherRad, radSum;
	double thisPos[MAXDIM], otherPos[MAXDIM];
	// zero out the force and get particle positions
	for (long dim = 0; dim < d_nDim; dim++) {
		pForce[particleId * d_nDim + dim] = 0;
		thisPos[dim] = pPos[particleId * d_nDim + dim];
	}
    thisRad = pRad[particleId];
    pEnergy[particleId] = 0;
    // interaction between vertices of neighbor particles
    for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
        radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					pEnergy[particleId] += calcContactInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::WCA:
					pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::adhesive:
					pEnergy[particleId] += calcAdhesiveInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
				}
				//if(particleId == 116 && d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId] == 109) printf("particleId %ld \t neighbor: %ld \t overlap %e \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum));
			}
    }
  }
}

// particle-wall interaction - across fictitious wall at half height in 2D
__global__ void kernelCalcParticleWallForce(const double* pRad, const double* pPos, const double range, double* wallForce) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		double thisRad, otherRad, radSum, midHeight = d_boxSizePtr[1]*0.5;
		double thisPos[MAXDIM], otherPos[MAXDIM], acrossForce[MAXDIM];
		double thisHeight, otherHeight;
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			acrossForce[dim] = 0;
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		wallForce[particleId] = 0;
		if(d_partMaxNeighborListPtr[particleId] > 5) {
			thisHeight = thisPos[1] - d_boxSizePtr[1] * floor(thisPos[1] / d_boxSizePtr[1]);
			if(thisHeight < midHeight && thisHeight > (midHeight - range)) {
				thisRad = pRad[particleId];
				// interaction between vertices of neighbor particles
				for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
					if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
						otherHeight = otherPos[1] - d_boxSizePtr[1] * floor(otherPos[1] / d_boxSizePtr[1]);
						if(otherHeight > midHeight && otherHeight < (midHeight + range)) {
							radSum = thisRad + otherRad;
							switch (d_simControl.potentialType) {
								case simControlStruct::potentialEnum::harmonic:
								calcContactInteraction(thisPos, otherPos, radSum, acrossForce);
								break;
								case simControlStruct::potentialEnum::lennardJones:
								calcLJInteraction(thisPos, otherPos, radSum, acrossForce);
								break;
								case simControlStruct::potentialEnum::WCA:
								calcWCAInteraction(thisPos, otherPos, radSum, acrossForce);
								break;
								case simControlStruct::potentialEnum::adhesive:
								calcAdhesiveInteraction(thisPos, otherPos, radSum, acrossForce);
								break;
							}
						//printf("particleId %ld otherId %ld \t thisHeight %lf \t otherHeight %lf \t wallForce %lf \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], thisHeight, otherHeight, acrossForce[1]);
						}
					}
				}
			}
			wallForce[particleId] += acrossForce[1];
			//printf("particleId %ld \t acrossForce %lf \t wallForce[particleId] %lf \n", particleId, acrossForce[1], wallForce[particleId]);
		}
	
  	}
}

// particle-sides contact interaction
__global__ void kernelCalcParticleSidesInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		double thisRad, radSum;
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		thisRad = pRad[particleId];
		radSum = 2 * thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		}
	}
}

// particle-box contact interaction
__global__ void kernelCalcParticleBoxInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		double thisRad, radSum;
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		thisRad = pRad[particleId];
		radSum = 2 * thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < thisRad) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		}
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		}
	}
}

// add gravity to particle force and energy
__global__ void kernelAddParticleGravity(const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		pForce[particleId * d_nDim + 1] -= d_gravity;
		pEnergy[particleId] += d_gravity * pPos[particleId * d_nDim + 1];
	}
}

__global__ void kernelCalcParticleStressTensor(const double* pRad, const double* pPos, double* pStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisRad, otherRad, radSum;
		double gradMultiple, distance;
		double thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], forces[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];
    // stress between neighbor particles
    for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      if(extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				radSum = thisRad + otherRad;
				gradMultiple = calcGradMultiple(thisPos, otherPos, radSum);
				if(gradMultiple > 0) {
					distance = calcDeltaAndDistance(thisPos, otherPos, delta);
					for (long dim = 0; dim < d_nDim; dim++) {
						forces[dim] = gradMultiple * delta[dim] / distance;
					}
					//diagonal terms
					pStress[0] = delta[0] * forces[0];
					pStress[3] = delta[1] * forces[1];
					// cross terms
					pStress[1] -= delta[0] * forces[1];
					pStress[2] -= delta[1] * forces[0];
				}
			}
		}
	}
}

inline __device__ void calcParticleStressFixedBox(const double* thisPos, const double* wallPos, const double radSum, double wallWork) {
  double overlap, gradMultiple, distance, distanceSq = 0.;
	double delta[MAXDIM], force[MAXDIM];
	for (long dim = 0; dim < d_nDim; dim++) {
		delta[dim] = thisPos[dim] - wallPos[dim];
		distanceSq += delta[dim] * delta[dim];
	}
	distance = sqrt(distanceSq);
	overlap = 1 - distance / radSum;
	if (overlap > 0) {
		gradMultiple = d_ec * overlap / radSum;
	  for (long dim = 0; dim < d_nDim; dim++) {
	    force[dim] = gradMultiple * delta[dim] / distance;
			wallWork -= force[dim] * thisPos[dim];
	  }
	}
}

__global__ void kernelCalcParticleBoxPressure(const double* pRad, const double* pPos, double wallWork) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		double thisRad, radSum;
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];
		radSum = thisRad;
		wallWork = 0;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < thisRad) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			calcParticleStressFixedBox(thisPos, wallPos, radSum, wallWork);
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			calcParticleStressFixedBox(thisPos, wallPos, radSum, wallWork);
		}
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			calcParticleStressFixedBox(thisPos, wallPos, radSum, wallWork);
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			calcParticleStressFixedBox(thisPos, wallPos, radSum, wallWork);
		}
	}
}

__global__ void kernelCalcParticleActivePressure(const double* pAngle, const double* pPos, const double driving, double activeWork) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double activeForce[MAXDIM], thisPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		for (long dim = 0; dim < d_nDim; dim++) {
      activeForce[dim] = driving * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId]));
			activeWork += activeForce[dim] * thisPos[dim];
    }
	}
}

//************************** neighbors and contacts **************************//
__global__ void kernelCalcParticleNeighborList(const double* pPos, const double* pRad, const double cutDistance) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    long addedNeighbor = 0;
    double thisRad, otherRad, radSum;
    double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];

    for (long otherId = 0; otherId < d_numParticles; otherId++) {
      if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
        bool isNeighbor = false;
        radSum = thisRad + otherRad;
        isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);
				//isNeighbor = (calcDistance(thisPos, otherPos) < cutDistance * radSum);
        if (addedNeighbor < d_partNeighborListSize) {
					d_partNeighborListPtr[particleId * d_partNeighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					//if(isNeighbor == true && particleId == 116) printf("particleId %ld \t otherId: %ld \t isNeighbor: %i \n", particleId, otherId, isNeighbor);
				}
				addedNeighbor += isNeighbor;
      }
    }
    d_partMaxNeighborListPtr[particleId] = addedNeighbor;
  }
}

__global__ void kernelCalcParticleBoxNeighborList(const double* pPos, const double* pRad, const double cutDistance) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    long addedNeighbor = 0;
    double thisRad, otherRad, radSum;
    double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];

    for (long otherId = 0; otherId < d_numParticles; otherId++) {
      if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
        bool isNeighbor = false;
        radSum = thisRad + otherRad;
        isNeighbor = (-calcFixedBoundaryOverlap(thisPos, otherPos, radSum) < cutDistance);
				//isNeighbor = (calcFixedBoundaryDistance(thisPos, otherPos) < cutDistance);
        if (addedNeighbor < d_partNeighborListSize) {
					d_partNeighborListPtr[particleId * d_partNeighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					//if(isNeighbor==true) printf("particleId %ld \t otherId: %ld \t overlap: %lf \n", particleId, otherId, calcOverlap(thisPos, otherPos, radSum));
				}
				addedNeighbor += isNeighbor;
      }
    }
    d_partMaxNeighborListPtr[particleId] = addedNeighbor;
  }
}

__global__ void kernelCalcParticleContacts(const double* pPos, const double* pRad, const double gapSize, const long contactLimit, long* contactList, long* numContacts) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    long addedContact = 0, newContactId;
    double thisRad, otherRad, radSum;
    double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];

		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				//if(particleId==0) printf("particleId %ld \t otherId: %ld \t overlap: %lf \n", particleId, particleId*d_partNeighborListSize + nListId, calcOverlap(thisPos, otherPos, radSum));
				radSum = thisRad + otherRad;
				if (calcOverlap(thisPos, otherPos, radSum) > (-gapSize)) {
					if (addedContact < contactLimit) {
						newContactId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
						bool isNewContact = true;
						for (long contactId = 0; contactId < contactLimit; contactId++) {
							if(newContactId == contactList[particleId * contactLimit + contactId]) {
								isNewContact = false;
							}
						}
						if(isNewContact) {
							contactList[particleId * contactLimit + addedContact] = newContactId;
							addedContact++;
						}
					}
				}
			}
			numContacts[particleId] = addedContact;
		}
	}
}

__global__ void kernelCalcContactVectorList(const double* pPos, const long* contactList, const long contactListSize, const long maxContacts, double* contactVectorList) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		for (long cListId = 0; cListId < maxContacts; cListId++) {
			long otherId = contactList[particleId * contactListSize + cListId];
			if ((particleId != otherId) && (otherId != -1)) {
				extractOtherParticlePos(particleId, otherId, pPos, otherPos);
				//Calculate the contactVector and put it into contactVectorList, which is a maxContacts*nDim by numParticle array
				calcDeltaAndDistance(thisPos, otherPos, &contactVectorList[particleId*(maxContacts*d_nDim) + cListId*d_nDim]);
			}
		}
	}
}

//******************************** observables *******************************//
__global__ void kernelCalcParticleDisplacement(const double* pPos, const double* pPreviousPos, double* pDisp) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		pDisp[particleId] = calcDistance(&pPos[particleId*d_nDim], &pPreviousPos[particleId*d_nDim]);
	}
}

__global__ void kernelCalcParticleDistanceSq(const double* pPos, const double* pInitialPos, double* pDelta) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double delta[MAXDIM];
		calcDeltaAndDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim], delta);
		for (long dim = 0; dim < d_nDim; dim++) {
			pDelta[particleId * d_nDim + dim] = delta[dim]*delta[dim];
		}
	}
}

__global__ void kernelCalcParticleScatteringFunction(const double* pPos, const double* pInitialPos, double* pSF, const double waveNum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double distance = 0;
		distance = calcDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim]);
		pSF[particleId] = sin(waveNum * distance) / (waveNum * distance);
	}
}

//******************************** integrators *******************************//
__global__ void kernelExtractThermalParticleVel(double* pVel, const double* r1, const double* r2, const double amplitude) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		double rNum[MAXDIM];
		rNum[0] = sqrt(-2.0 * log(r1[particleId])) * cos(2.0 * PI * r2[particleId]);
		rNum[1] = sqrt(-2.0 * log(r1[particleId])) * sin(2.0 * PI * r2[particleId]);
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = amplitude * rNum[dim];
		}
  }
}

__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] += timeStep * pVel[particleId * d_nDim + dim];
		}
  }
}

__global__ void kernelCheckParticlePBC(double* pPosPBC, const double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pPosPBC[particleId * d_nDim + dim] = pPos[particleId * d_nDim + dim] - floor(pPos[particleId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
		}
	}
}

__global__ void kernelUpdateParticleVel(double* pVel, const double* pForce, const double timeStep) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
    for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] += timeStep * pForce[particleId * d_nDim + dim];
		}
  }
}

__global__ void kernelUpdateBrownianParticleVel(double* pVel, const double* pForce, double* thermalVel, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = mobility * pForce[particleId * d_nDim + dim] + thermalVel[particleId * d_nDim + dim];
		}
  }
}

__global__ void kernelUpdateActiveParticleVel(double* pVel, const double* pForce, double* pAngle, const double driving, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		double angle = pAngle[particleId];
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = mobility * (pForce[particleId * d_nDim + dim] + driving * ((1 - dim) * cos(angle) + dim * sin(angle)));
		}
  }
}

__global__ void kernelConserveParticleMomentum(double* pVel) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double COMP[MAXDIM];
  if (threadIdx.x == 0) {
    for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] = 0.0;
		}
  }
  __syncthreads();

  if (particleId < d_numParticles) {
    for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&COMP[dim], pVel[particleId * d_nDim + dim]);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] /= d_numParticles;
		}
  }
  __syncthreads();

  if (particleId < d_numParticles) {
    for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= COMP[dim];
		}
  }
}

__global__ void kernelConserveSubSetMomentum(double* pVel, const long firstId) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double COMP[MAXDIM];
  if (threadIdx.x == 0) {
    for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] = 0.0;
		}
  }
  __syncthreads();

  if (particleId < d_numParticles && particleId > firstId) {
    for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&COMP[dim], pVel[particleId * d_nDim + dim]);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] /= (d_numParticles - firstId);
		}
  }
  __syncthreads();

  if (particleId < d_numParticles && particleId > firstId) {
    for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= COMP[dim];
		}
  }
}

__global__ void kernelSumParticleVelocity(const double* pVel, double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&velSum[dim], pVel[particleId * d_nDim + dim]);
		}
	}
}

__global__ void kernelSubtractParticleDrift(double* pVel, const double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&pVel[particleId * d_nDim + dim], -velSum[dim]/d_numParticles);
		}
	}
}


#endif /* DPM2DKERNEL_CUH_ */
