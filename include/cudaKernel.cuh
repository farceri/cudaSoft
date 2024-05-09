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
// energy costant
__constant__ double d_ec;
// adhesive constants
__constant__ double d_l1;
__constant__ double d_l2;
// Lennard-Jones constants
__constant__ double d_LJcutoff;
__constant__ double d_LJecut;
__constant__ double d_LJfshift;
__constant__ double d_LJecutPlus;
__constant__ double d_LJfshiftPlus;
// Double Lennard-Jones
__constant__ double d_eAA;
__constant__ double d_eAB;
__constant__ double d_eBB;
__constant__ long d_num1;
// Mie constants
__constant__ double d_nPower;
__constant__ double d_mPower;
__constant__ double d_mieConstant;
__constant__ double d_Miecut;
// Gravity
__constant__ double d_gravity;
__constant__ double d_ew; // wall
// Fluid flow
__constant__ double d_flowSpeed;
__constant__ double d_flowDecay;
__constant__ double d_flowViscosity;

// particle neighborList
__constant__ long* d_partNeighborListPtr;
__constant__ long* d_partMaxNeighborListPtr;
__constant__ long d_partNeighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_partMaxNeighbors;


inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	double delta = x1 - x2, size = d_boxSizePtr[dim];
	//if (2*delta < -size) return delta + size;
	//if (2*delta > size) return delta - size;
	return delta - size * round(delta / size); //round for distance, floor for position
}

//for leesEdwards need to handle first two dimensions together
inline __device__ double pbcDistanceLE(const double x1, const double y1, const double x2, const double y2) {
	auto deltax = (x1 - x2);
	auto rounded = round(deltax); //need to store for lees-edwards BC
	deltax -= rounded;
	auto deltay = (y1 - y2);
	deltay = deltay - rounded*d_LEshift - round(deltay - rounded*d_LEshift);
	return deltay;
}

inline __device__ double calcNorm(const double* segment) {
  auto normSq = 0.0;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return sqrt(normSq);
}

inline __device__ double calcNormSq(const double* segment) {
  auto normSq = 0.0;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return normSq;
}

inline __device__ double calcDistance(const double* thisVec, const double* otherVec) {
  auto distanceSq = 0.0;
  auto delta = 0.0;
  double deltay, deltax, shifty;
	switch (d_simControl.geometryType) {
		case simControlStruct::geometryEnum::normal:
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::leesEdwards:
		deltay = thisVec[1] - otherVec[1];
		shifty = round(deltay / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltax = thisVec[0] - otherVec[0];
		deltax -= shifty * d_LEshift;
		deltax -= round(deltax / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltay -= shifty;
		distanceSq = deltax * deltax + deltay * deltay;
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedBox:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
		delta = thisVec[1] - otherVec[1];
		distanceSq = delta * delta;
		delta = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq += delta * delta;
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedSides3D:
		delta = thisVec[2] - otherVec[2];
		distanceSq = delta * delta;
		delta = pbcDistance(thisVec[1], otherVec[1], 1);
		distanceSq += delta * delta;
		delta = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq += delta * delta;
		return sqrt(distanceSq);
		break;
		default:
		return 0;
		break;
	}
}

inline __device__ void calcDelta(const double* thisVec, const double* otherVec, double* deltaVec) {
	auto delta = 0.0;
	auto shifty = 0.0;
	switch (d_simControl.geometryType) {
		case simControlStruct::geometryEnum::normal:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
	    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			deltaVec[dim] = delta;
	 	}
		break;
		case simControlStruct::geometryEnum::leesEdwards:
		deltaVec[1] = thisVec[1] - otherVec[1];
		shifty = round(deltaVec[1] / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltaVec[0] = thisVec[0] - otherVec[0];
		deltaVec[0] -= shifty * d_LEshift;
		deltaVec[0] -= round(deltaVec[0] / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltaVec[1] -= shifty;
		break;
		case simControlStruct::geometryEnum::fixedBox:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			deltaVec[dim] = delta;
		}
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
		deltaVec[1] = thisVec[1] - otherVec[1];
		deltaVec[0] = pbcDistance(thisVec[0], otherVec[0], 0);
		break;
		case simControlStruct::geometryEnum::fixedSides3D:
		deltaVec[2] = thisVec[2] - otherVec[2];
		deltaVec[1] = pbcDistance(thisVec[1], otherVec[1], 1);
		deltaVec[0] = pbcDistance(thisVec[0], otherVec[0], 0);
		break;
		default:
		break;
	}
}

inline __device__ double calcDeltaAndDistance(const double* thisVec, const double* otherVec, double* deltaVec) {
	auto distanceSq = 0.0;
	auto delta = 0.0;
	auto shifty = 0.0;
	switch (d_simControl.geometryType) {
		case simControlStruct::geometryEnum::normal:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
	    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			deltaVec[dim] = delta;
	    	distanceSq += delta * delta;
	 	}
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::leesEdwards:
		deltaVec[1] = thisVec[1] - otherVec[1];
		shifty = round(deltaVec[1] / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltaVec[0] = thisVec[0] - otherVec[0];
		deltaVec[0] -= shifty * d_LEshift;
		deltaVec[0] -= round(deltaVec[0] / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltaVec[1] -= shifty;
		distanceSq = deltaVec[0] * deltaVec[0] + deltaVec[1] * deltaVec[1];
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedBox:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			deltaVec[dim] = delta;
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedSides2D:
		deltaVec[1] = thisVec[1] - otherVec[1];
		distanceSq = deltaVec[1] * deltaVec[1];
		deltaVec[0] = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq += deltaVec[0] * deltaVec[0];
		return sqrt(distanceSq);
		break;
		case simControlStruct::geometryEnum::fixedSides3D:
		deltaVec[2] = thisVec[2] - otherVec[2];
		distanceSq = deltaVec[2] * deltaVec[2];
		deltaVec[1] = pbcDistance(thisVec[1], otherVec[1], 1);
		distanceSq += deltaVec[1] * deltaVec[1];
		deltaVec[0] = pbcDistance(thisVec[0], otherVec[0], 0);
		distanceSq += deltaVec[0] * deltaVec[0];
		return sqrt(distanceSq);
		break;
		default:
		return 0;
		break;
	}
}

inline __device__ double calcFixedBoundaryDistance(const double* thisVec, const double* otherVec) {
  	auto distanceSq = 0.0;
  	auto delta = 0.0;
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
	if (particleId != otherId) {
		getParticlePos(otherId, pPos, otherPos);
		otherRad = pRad[otherId];
    	return true;
  	}
  	return false;
}

inline __device__ bool extractOtherParticlePos(const long particleId, const long otherId, const double* pPos, double* otherPos) {
	if (particleId != otherId) {
		getParticlePos(otherId, pPos, otherPos);
    	return true;
  	}
  	return false;
}

inline __device__ bool extractParticleNeighbor(const long particleId, const long nListId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	auto otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
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
	auto otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
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

inline __device__ double calcLJForceShift(const double radSum) {
	auto ratio6 = pow(d_LJcutoff, 6);
	return 24 * d_ec * (2 / ratio6 - 1) / (d_LJcutoff * radSum * ratio6);
}

inline __device__ double calcDoubleLJForceShift(const double epsilon, const double radSum) {
	auto ratio6 = pow(d_LJcutoff, 6);
	return 24 * epsilon * (2 / ratio6 - 1) / (d_LJcutoff * radSum * ratio6);
}

inline __device__ double calcMieForceShift(const double radSum) {
	auto nRatio = pow(d_LJcutoff, d_nPower);
	auto mRatio = pow(d_LJcutoff, d_mPower);
	return d_mieConstant * d_ec * (d_nPower / nRatio - d_mPower / mRatio) / (d_LJcutoff * radSum);
}

inline __device__ double calcGradMultiple(const long particleId, const long otherId, const double* thisPos, const double* otherPos, const double radSum) {
	auto distance = calcDistance(thisPos, otherPos);
	double overlap, ratio, ratio6, ratio12, ration, ratiom, forceShift, gradMultiple = 0.0;
	double sign = -1.0;
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
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (d_LJcutoff * radSum)) {
			forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
			return 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::Mie:
		ratio = radSum / distance;
		ration = pow(ratio, d_nPower);
		ratiom = pow(ratio, d_mPower);
		if (distance < (d_LJcutoff * radSum)) {
			forceShift = calcMieForceShift(radSum);
			return d_mieConstant * d_ec * (d_nPower * ration - d_mPower * ratiom) / distance - forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::WCA:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (WCAcut * radSum)) {
			return 24 * d_ec * (2 * ratio12 - ratio6) / distance;
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
		case simControlStruct::potentialEnum::doubleLJ:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (d_LJcutoff * radSum)) {
			forceShift = d_LJfshift / radSum;//calcDoubleLJForceShift(epsilon, radSum);
			gradMultiple = 24 * (2 * ratio12 - ratio6) / distance - forceShift;
			if(particleId < d_num1) {
				if(otherId < d_num1) {
					gradMultiple *= d_eAA;
				} else {
					gradMultiple *= d_eAB;
				}
			} else {
				if(otherId >= d_num1) {
					gradMultiple *= d_eBB;
				} else {
					gradMultiple *= d_eAB;
				}
			}
			return gradMultiple;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::LJMinusPlus:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (d_LJcutoff * radSum)) {
			forceShift = d_LJfshift / radSum;
			if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
				sign = 1.0;
				forceShift = d_LJfshiftPlus / radSum;
			}
			return 24 * d_ec * (2 * ratio12 + sign * ratio6) / distance - forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::LJWCA:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if(particleId < d_num1 && otherId < d_num1) {
			if (distance < (d_LJcutoff * radSum)) {
				forceShift =  d_LJfshift / radSum;
				return 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
			} else {
				return 0;
			}
		} else if(particleId >= d_num1 && otherId >= d_num1) {
			if (distance < (d_LJcutoff * radSum)) {
				forceShift =  d_LJfshift / radSum;
				return 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
			} else {
				return 0;
			}
		} else {
			if (distance < (WCAcut * radSum)) {
				return 24 * d_ec * (2 * ratio12 - ratio6) / distance;
			} else {
				return 0;
			}
		}
		break;
		default:
		return 0;
		break;
	}
}

inline __device__ double calcContactInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	//calcDelta(thisPos, otherPos, delta);
	//distance = calcNorm(delta);
	//overlap = calcOverlap(thisPos, otherPos, radSum);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ec * overlap / radSum;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentForce[dim] += gradMultiple * delta[dim] / distance;
		}
	  	return (0.5 * d_ec * overlap * overlap) * 0.5;
	}
	return 0.;
}

inline __device__ double calcWallContactInteraction(const double* thisPos, const double* wallPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	//overlap = calcOverlap(thisPos, wallPos, radSum);
	auto distance = calcDeltaAndDistance(thisPos, wallPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ew * overlap / radSum;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentForce[dim] += gradMultiple * delta[dim] / distance;
		}
		return (0.5 * d_ew * overlap * overlap);
	}
	return 0.;
}

inline __device__ double calcWallWCAInteraction(const double* thisPos, const double* wallPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, wallPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (WCAcut * radSum)) {
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
	  	return 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
	}
	return 0.0;
}

inline __device__ double calcLJInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
	} else {
		return 0.0;
	}
}

inline __device__ double calcWCAInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (WCAcut * radSum)) {
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	 	}
	  	return 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
	} else {
		return 0.0;
	}
}

inline __device__ double calcMieInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ration = pow(ratio, d_nPower);
	auto ratiom = pow(ratio, d_mPower);
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift = calcMieForceShift(radSum);
		auto gradMultiple =  d_mieConstant * d_ec * (d_nPower * ration - d_mPower * ratiom) / distance - forceShift;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return 0.5 * (d_mieConstant * d_ec * ((ration - ratiom) - d_Miecut) - abs(forceShift) * (distance - d_LJcutoff * radSum));
	} else {
		return 0.0;
	}
}

inline __device__ double calcAdhesiveInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  	auto gradMultiple = 0.0;
	auto epot = 0.0;
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if (distance < (1 + d_l1) * radSum) {
		gradMultiple = d_ec * overlap / radSum;
		epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
	} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
		gradMultiple = -d_ec * (d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		epot = -0.5 * d_ec * (d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2);
	} else {
		return 0.0;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

inline __device__ double calcDoubleLJInteraction(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId, double* currentForce) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift = d_LJfshift / radSum;//calcDoubleLJForceShift(epsilon, radSum);
		auto gradMultiple = 24 * (2 * ratio12 - ratio6) / distance - forceShift;
		auto epot = 0.5 * (4 * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
		// set energy scale based on particle indices
		if(particleId < d_num1) {
			if(otherId < d_num1) {
				gradMultiple *= d_eAA;
				epot *= d_eAA;
			} else {
				gradMultiple *= d_eAB;
				epot *= d_eAB;
			}
		} else {
			if(otherId >= d_num1) {
				gradMultiple *= d_eBB;
				epot *= d_eBB;
			} else {
				gradMultiple *= d_eAB;
				epot *= d_eAB;
			}
		}
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return epot;
	} else {
		return 0.0;
	}
}

inline __device__ double calcLJMinusPlusInteraction(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId, double* currentForce) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto sign = -1.0;
		auto forceShift = d_LJfshift / radSum;
		auto ecut = d_LJecut;
		auto multiple = 1;
		if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
			//printf("particleId %ld otherId %ld d_num1: %ld\n", particleId, otherId, d_num1);
			sign = 1.0;
			forceShift = d_LJfshiftPlus / radSum;
			ecut = d_LJecutPlus;
		}
		auto gradMultiple = multiple * 24 * d_ec * (2 * ratio12 + sign * ratio6) / distance - forceShift;
		auto epot = multiple * 0.5 * (4 * d_ec * (ratio12 + sign * ratio6) - ecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return epot;
	} else {
		return 0.0;
	}
}

// particle-particle interaction
__global__ void kernelCalcParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto otherId = -1;
    	double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0;
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		auto thisRad = pRad[particleId];
		pEnergy[particleId] = 0;
		// interaction between vertices of neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					pEnergy[particleId] += calcContactInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::Mie:
					pEnergy[particleId] += calcMieInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::WCA:
					pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::doubleLJ:
					otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
					pEnergy[particleId] += calcDoubleLJInteraction(thisPos, otherPos, radSum, particleId, otherId, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::LJMinusPlus:
					otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
					pEnergy[particleId] += calcLJMinusPlusInteraction(thisPos, otherPos, radSum, particleId, otherId, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::LJWCA:
					otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
					if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
						pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					} else {
						pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					}
					break;
					default:
					break;
				}
				//if(particleId == 116 && d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId] == 109) printf("particleId %ld \t neighbor: %ld \t overlap %e \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum));
			}
		}
  	}
}

__global__ void kernelCalcAllToAllParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		//printf("particleId %ld\n", particleId);
    	double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0;
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		auto thisRad = pRad[particleId];
		pEnergy[particleId] = 0;
		// interaction between vertices of neighbor particles
		for (long otherId = 0; otherId < d_numParticles; otherId++) {
			if (extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				//printf("numParticles: %ld otherId %ld particleId %ld radSum %lf\n", d_numParticles, otherId, particleId, radSum);
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					pEnergy[particleId] += calcContactInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::Mie:
					pEnergy[particleId] += calcMieInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::WCA:
					pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::doubleLJ:
					pEnergy[particleId] += calcDoubleLJInteraction(thisPos, otherPos, radSum, particleId, otherId, &pForce[particleId*d_nDim]);
					//printf("particleId %ld otherId %ld energy %lf \n", particleId, otherId, pEnergy[particleId]);
					break;
					case simControlStruct::potentialEnum::LJWCA:
					if(particleId < d_num1 && otherId < d_num1) {
						pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					} else if(particleId >= d_num1 && otherId >= d_num1) {
						pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					} else {
						pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					}
					default:
					break;
				}
				//if(pEnergy[particleId] != 0) printf("particleId %ld otherId %ld pForce[particleId] %e %e\n", particleId, otherId, pForce[particleId * d_nDim], pForce[particleId * d_nDim + 1]);
				//if(particleId == 116 && d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId] == 109) printf("particleId %ld \t neighbor: %ld \t overlap %e \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum));
			}
		}
  	}
}

inline __device__ double calcContactYforce(const double* thisPos, const double* otherPos, const double radSum) {
	double delta[MAXDIM];
	//overlap = calcOverlap(thisPos, otherPos, radSum);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		return (d_ec * overlap / radSum) * delta[1] / distance;
	} else {
		return 0;
	}
}

inline __device__ double calcLJYforce(const double* thisPos, const double* otherPos, const double radSum) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
		return (24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift) * delta[1] / distance;
	} else {
		return 0;
	}
}

inline __device__ double calcWCAYforce(const double* thisPos, const double* otherPos, const double radSum) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (WCAcut * radSum)) {
		return (24 * d_ec * (2 * ratio12 - ratio6) / distance) * delta[1] / distance;
	} else {
		return 0;
	}
}

inline __device__ double calcDoubleLJYforce(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift = d_LJfshift / radSum;//calcDoubleLJForceShift(epsilon, radSum);
		auto gradMultiple = 24 * (2 * ratio12 - ratio6) / distance - forceShift;
		// set energy scale based on particle indices
		if(particleId < d_num1) {
			if(otherId < d_num1) {
				gradMultiple *= d_eAA;
			} else {
				gradMultiple *= d_eAB;
			}
		} else {
			if(otherId >= d_num1) {
				gradMultiple *= d_eBB;
			} else {
				gradMultiple *= d_eAB;
			}
		}
		return gradMultiple * delta[1] / distance;
	} else {
		return 0;
	}
}

inline __device__ double calcLJMinusPlusYforce(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto sign = -1.0;
		auto forceShift = d_LJfshift / radSum;
		if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
			sign = 1.0;
			forceShift = d_LJfshiftPlus / radSum;
		}
		auto gradMultiple = 24 * d_ec * (2 * ratio12 + sign * ratio6) / distance - forceShift;
		return gradMultiple * delta[1] / distance;
	} else {
		return 0.0;
	}
}


// particle-particle interaction across fictitious wall at half height in 2D
__global__ void kernelCalcParticleWallForce(const double* pRad, const double* pPosPBC, const double range, double* wallForce, long* wallCount) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		long otherId;
		auto midHeight = d_boxSizePtr[1]*0.5;
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPosPBC[particleId * d_nDim + dim];
		}
		wallForce[particleId] = 0;
		wallCount[particleId] = 0;
		if(d_partMaxNeighborListPtr[particleId] > 0) {
			auto thisRad = pRad[particleId];
			auto thisHeight = thisPos[1];// - d_boxSizePtr[1] * floor(thisPos[1] / d_boxSizePtr[1]);
			//thisDistance = thisPos[1] - midHeight;
			//if(thisDistance < 0) {
			if(thisHeight < midHeight && thisHeight > (midHeight - range)) {
				// interaction between vertices of neighbor particles
				for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
					if (extractParticleNeighbor(particleId, nListId, pPosPBC, pRad, otherPos, otherRad)) {
						auto otherHeight = otherPos[1];// - d_boxSizePtr[1] * floor(otherPos[1] / d_boxSizePtr[1]);
						//otherDistance = otherPos[1] - midHeight;
						//if(otherDistance > 0) {
						if(otherHeight > midHeight && otherHeight < (midHeight + range)) {
							wallCount[particleId] += 1;
							auto radSum = thisRad + otherRad;
							switch (d_simControl.potentialType) {
								case simControlStruct::potentialEnum::harmonic:
								wallForce[particleId] += calcContactYforce(thisPos, otherPos, radSum);
								break;
								case simControlStruct::potentialEnum::lennardJones:
								wallForce[particleId] += calcLJYforce(thisPos, otherPos, radSum);
								break;
								case simControlStruct::potentialEnum::WCA:
								wallForce[particleId] += calcWCAYforce(thisPos, otherPos, radSum);
								break;
								case simControlStruct::potentialEnum::doubleLJ:
								otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
								wallForce[particleId] += calcDoubleLJYforce(thisPos, otherPos, radSum, particleId, otherId);
								break;
								case simControlStruct::potentialEnum::LJMinusPlus:
								otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
								wallForce[particleId] += calcLJMinusPlusYforce(thisPos, otherPos, radSum, particleId, otherId);
								break;
								case simControlStruct::potentialEnum::LJWCA:
								otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
								if(particleId < d_num1 && otherId < d_num1) {
									wallForce[particleId] += calcLJYforce(thisPos, otherPos, radSum);
								} else if(particleId >= d_num1 && otherId >= d_num1) {
									wallForce[particleId] += calcLJYforce(thisPos, otherPos, radSum);
								} else {
									wallForce[particleId] += calcWCAYforce(thisPos, otherPos, radSum);
								}
								default:
								break;
							}
						//printf("particleId %ld otherId %ld \t thisHeight %lf \t otherHeight %lf \t wallForce %lf \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], thisHeight, otherHeight, acrossForce[1]);
						}
					}
				}
			}
			//printf("particleId %ld \t acrossForce %lf \t wallForce[particleId] %lf \n", particleId, acrossForce[1], wallForce[particleId]);
		}
  	}
}

// particle-particle interaction across fictitious wall at half height in 2D
__global__ void kernelAddParticleWallActiveForce(const double* pAngle, const double driving, double* wallForce, long* wallCount) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		// if the interaction of particleId has already been counted
		if(wallCount[particleId] > 0) {
			wallForce[particleId] += driving * sin(pAngle[particleId]);
		}
  	}
}

// particle-sides contact interaction in 2D
__global__ void kernelCalcParticleSidesInteraction2D(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
	}
}

// particle-sides contact interaction in 3D
__global__ void kernelCalcParticleSidesInteraction3D(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[2] < thisRad) {
			wallPos[2] = 0;
			wallPos[1] = thisPos[1];
			wallPos[0] = thisPos[0];
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
		} else if((d_boxSizePtr[2] - thisPos[2]) < thisRad) {
			wallPos[2] = d_boxSizePtr[2];
			wallPos[1] = thisPos[1];
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
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < thisRad) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			case simControlStruct::boxEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
	}
}

// add gravity to particle force and energy
__global__ void kernelAddParticleGravity(const double* pPos, double* pForce, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		pForce[particleId * d_nDim + d_nDim - 1] -= d_gravity;
		pEnergy[particleId] += d_gravity * pPos[particleId * d_nDim + d_nDim - 1];
	}
}

// compute particle-dependent surface height for fluid flow
__global__ void kernelCalcSurfaceHeight(const double* pPos, const long* numContacts, double* sHeight) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		if (numContacts[particleId] >= 1) {
			sHeight[particleId] = pPos[particleId * d_nDim + d_nDim - 1];
		} else {
			sHeight[particleId] = 0;
		}
	}
}

// compute flow velocity with law of the wall
__global__ void kernelCalcFlowVelocity(const double* pPos, const double* sHeight, double* flowVel) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			flowVel[particleId * d_nDim + dim] = 0;
		}
		//flowVel[particleId * d_nDim + 1] = d_flowSpeed * pPos[particleId * d_nDim + 1] / d_boxSizePtr[1];
		if(pPos[particleId * d_nDim + 1] >= sHeight[particleId]) {
			flowVel[particleId * d_nDim] = d_flowSpeed * (log(1 + (pPos[particleId * d_nDim + d_nDim - 1] - sHeight[particleId]) * d_flowSpeed / d_flowViscosity) + 1);
		} else {
			flowVel[particleId * d_nDim] = d_flowSpeed / exp(d_flowDecay * (sHeight[particleId] - pPos[particleId * d_nDim + d_nDim - 1]));
		}
	}
}

__global__ void kernelCalcParticleStressTensor(const double* pRad, const double* pPos, const double* pVel, double* pStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], forces[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		// thermal stress
		pStress[0] += pVel[particleId * d_nDim] * pVel[particleId * d_nDim + 1];
		pStress[3] += pVel[particleId * d_nDim + 1] * pVel[particleId * d_nDim + 1];
		// cross terms
		pStress[1] += pVel[particleId * d_nDim] * pVel[particleId * d_nDim + 1];
		pStress[2] += pVel[particleId * d_nDim + 1] * pVel[particleId * d_nDim];
		// stress between neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
			if(extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				auto gradMultiple = calcGradMultiple(particleId, otherId, thisPos, otherPos, radSum);
				if(gradMultiple > 0) {
					auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
					for (long dim = 0; dim < d_nDim; dim++) {
						forces[dim] = gradMultiple * delta[dim] / distance;
					}
					//diagonal terms
					pStress[0] += delta[0] * forces[0];
					pStress[3] += delta[1] * forces[1];
					// cross terms
					pStress[1] += delta[0] * forces[1];
					pStress[2] += delta[1] * forces[0];
				}
			}
		}
	}
}

inline __device__ void calcWallContactStress(const double* thisPos, const double* wallPos, const double radSum, double* wallStress) {
  	auto distanceSq = 0.0;
	double delta[MAXDIM], force[MAXDIM];
	for (long dim = 0; dim < d_nDim; dim++) {
		delta[dim] = thisPos[dim] - wallPos[dim];
		distanceSq += delta[dim] * delta[dim];
	}
	auto distance = sqrt(distanceSq);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ew * overlap / radSum;
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
	    	force[dim] = gradMultiple * delta[dim] / distance;
			wallStress[dim * d_nDim + dim] += force[dim] * delta[dim];
	  	}
	}
}

inline __device__ void calcWallWCAStress(const double* thisPos, const double* wallPos, const double radSum, double* wallStress) {
	auto distanceSq = 0.0;
	double delta[MAXDIM], force[MAXDIM];
	for (long dim = 0; dim < d_nDim; dim++) {
		delta[dim] = thisPos[dim] - wallPos[dim];
		distanceSq += delta[dim] * delta[dim];
	}
	auto distance = sqrt(distanceSq);
	auto ratio = radSum / distance;
	auto ratio12 = pow(ratio, 12);
	auto ratio6 = pow(ratio, 6);
	if (distance <= (WCAcut * radSum)) {
		auto gradMultiple = 4 * d_ew * (12 * ratio12 - 6 * ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	force[dim] += gradMultiple * delta[dim] / distance;
			wallStress[dim * d_nDim + dim] += force[dim] * delta[dim];
	  	}
	}
}

__global__ void kernelCalcParticleBoxStress(const double* pRad, const double* pPos, double* wallStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    	auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < thisRad) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
		}
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
		}
	}
}

__global__ void kernelCalcParticleSides2DStress(const double* pRad, const double* pPos, double* wallStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    	auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[1] < thisRad) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.boxType) {
			case simControlStruct::boxEnum::harmonic:
			calcWallContactStress(thisPos, wallPos, radSum, wallStress);
			break;
			case simControlStruct::boxEnum::WCA:
			calcWallWCAStress(thisPos, wallPos, radSum, wallStress);
			break;
			default:
			break;
			}
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
		auto addedNeighbor = 0;
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		for (long otherId = 0; otherId < d_numParticles; otherId++) {
			if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
				bool isNeighbor = false;
				auto radSum = thisRad + otherRad;
				isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);// cutDistance should be greater than radSum
				//isNeighbor = (calcDistance(thisPos, otherPos) < (cutDistance * radSum));
				if (addedNeighbor < d_partNeighborListSize) {
					d_partNeighborListPtr[particleId * d_partNeighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					//if(isNeighbor == true) printf("particleId %ld \t otherId: %ld \t isNeighbor: %i \n", particleId, otherId, isNeighbor);
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
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		for (long otherId = 0; otherId < d_numParticles; otherId++) {
			if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
				bool isNeighbor = false;
				auto radSum = thisRad + otherRad;
				//isNeighbor = (-calcFixedBoundaryOverlap(thisPos, otherPos, radSum) < cutDistance);
				isNeighbor = (calcFixedBoundaryDistance(thisPos, otherPos) < (cutDistance * radSum));
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
   		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    	auto thisRad = pRad[particleId];

		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				//if(particleId==0) printf("particleId %ld \t otherId: %ld \t overlap: %lf \n", particleId, particleId*d_partNeighborListSize + nListId, calcOverlap(thisPos, otherPos, radSum));
				auto radSum = thisRad + otherRad;
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
			auto otherId = contactList[particleId * contactListSize + cListId];
			if ((particleId != otherId) && (otherId != -1)) {
				extractOtherParticlePos(particleId, otherId, pPos, otherPos);
				//Calculate the contactVector and put it into contactVectorList, which is a maxContacts*nDim by numParticle array
				calcDeltaAndDistance(thisPos, otherPos, &contactVectorList[particleId*(maxContacts*d_nDim) + cListId*d_nDim]);
			}
		}
	}
}

//******************************** observables *******************************//
__global__ void kernelCalcParticleVelSquared(const double* pVel, double* velSq) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		velSq[particleId] = 0;
		for (long dim = 0; dim < d_nDim; dim++) {
			velSq[particleId] += pVel[particleId * d_nDim + dim] * pVel[particleId * d_nDim + dim];
		}
	}
}

__global__ void kernelCalcParticleDisplacement(const double* pPos, const double* pLastPos, double* pDisp) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		pDisp[particleId] = calcDistance(&pPos[particleId*d_nDim], &pLastPos[particleId*d_nDim]);
	}
}

__global__ void kernelCheckParticleDisplacement(const double* pPos, const double* pLastPos, int* flag, double cutoff) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto displacement = calcDistance(&pPos[particleId*d_nDim], &pLastPos[particleId*d_nDim]);
		if(2 * displacement > cutoff) {
			flag[particleId] = 1;
		}
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
		auto distance = calcDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim]);
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

__global__ void kernelSetPBC(double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] -= floor(pPos[particleId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
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
		auto angle = pAngle[particleId];
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
