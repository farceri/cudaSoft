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

//box radius for circular geometry
__constant__ double d_boxRadius;

__constant__ long d_nDim;
__constant__ long d_numParticles;

// time step
__constant__ double d_dt;
// dimensionality factor
__constant__ double d_rho0;
// energy costant
__constant__ double d_ec;
// activity constants
__constant__ double d_driving;
__constant__ double d_taup;
// Vicsek interaction constant
__constant__ double d_Jvicsek;
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
// Wall shape constants
__constant__ long d_numWall;
__constant__ double d_wallRad;
__constant__ double d_wallArea;
__constant__ double d_wallArea0;
__constant__ double d_wallLength0;
__constant__ double d_wallAngle0;
__constant__ double d_wallEnergy;
__constant__ double d_ea;
__constant__ double d_eb;
__constant__ double d_el;
__constant__ double d_stiff = 1;
__constant__ double d_extSq = 1;
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

// particle-wall interaction neighborList
__constant__ long* d_wallNeighborListPtr;
__constant__ long* d_wallMaxNeighborListPtr;
__constant__ long d_wallNeighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_wallMaxNeighbors;

// vicsek interaction neighborList
__constant__ long* d_vicsekNeighborListPtr;
__constant__ long* d_vicsekMaxNeighborListPtr;
__constant__ long d_vicsekNeighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_vicsekMaxNeighbors;


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

inline __device__ void checkAngleMinusPIPlusPI(double &angle) {
	angle = angle + PI;
	angle = angle - 2.0 * PI * floor(angle / (2.0 * PI));
	angle = angle - PI;
}

inline __device__ void cartesianToPolar(const double* vec, double &r, double &theta) {
    // Compute radial distance and angle wrt the origin
    r = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    theta = atan2(vec[1], vec[0]);
}

inline __device__ void polarToCartesian(double* vec, const double r, const double theta) {
    // Compute cartesian coordinates from radial distance and angle
	vec[0] = r * cos(theta);
	vec[1] = r * sin(theta);
}

inline __device__ double calcDeltaAngle(double thisTheta, double otherTheta) {
    // Ensure angular difference is within [-π, π]
	auto deltaTheta = thisTheta - otherTheta;
    checkAngleMinusPIPlusPI(deltaTheta);
    return deltaTheta;
}

inline __device__ double pbcDistanceRound(const double thisR, const double thisTheta, const double otherR, const double otherTheta) {
    // Compute the angular difference with periodic boundary conditions
    double deltaTheta = calcDeltaAngle(thisTheta, otherTheta);
    // Compute Euclidean distance using polar coordinates
    return sqrt(thisR * thisR + otherR * otherR - 2 * thisR * otherR * cos(deltaTheta));
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

inline __device__ void getDelta(const double* thisVec, const double* otherVec, double* delta) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta[dim] = thisVec[dim] - otherVec[dim];
  	}
}

inline __device__ double calcDistance(const double* thisVec, const double* otherVec) {
	auto distanceSq = 0.0;
  	auto delta = 0.0;
  	double deltay, deltax, shifty;
  	//double r1, theta1, r2, theta2;
  	switch (d_simControl.boundaryType) {
		case simControlStruct::boundaryEnum::pbc:
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
		case simControlStruct::boundaryEnum::leesEdwards:
		deltay = thisVec[1] - otherVec[1];
		shifty = round(deltay / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltax = thisVec[0] - otherVec[0];
		deltax -= shifty * d_LEshift;
		deltax -= round(deltax / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltay -= shifty;
		distanceSq = deltax * deltax + deltay * deltay;
		return sqrt(distanceSq);
		break;
		case simControlStruct::boundaryEnum::fixed:
		switch (d_simControl.geometryType) {
			case simControlStruct::geometryEnum::squareWall:
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
			case simControlStruct::geometryEnum::roundWall:
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < d_nDim; dim++) {
				delta = thisVec[dim] - otherVec[dim];
				distanceSq += delta * delta;
			}
			return sqrt(distanceSq);
			break;
		}
		break;
		default: // reflect, reflectNoise, rigid and mobile
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
	}
}

inline __device__ double calcDeltaAndDistance(const double* thisVec, const double* otherVec, double* deltaVec) {
	auto distanceSq = 0.0;
	auto delta = 0.0;
	auto shifty = 0.0;
  	//double r1, theta1, r2, theta2, deltaR, deltaTheta;
	switch (d_simControl.boundaryType) {
		case simControlStruct::boundaryEnum::pbc:
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
	    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
			deltaVec[dim] = delta;
	    	distanceSq += delta * delta;
	 	}
		return sqrt(distanceSq);
		break;
		case simControlStruct::boundaryEnum::leesEdwards:
		deltaVec[1] = thisVec[1] - otherVec[1];
		shifty = round(deltaVec[1] / d_boxSizePtr[1]) * d_boxSizePtr[1];
		deltaVec[0] = thisVec[0] - otherVec[0];
		deltaVec[0] -= shifty * d_LEshift;
		deltaVec[0] -= round(deltaVec[0] / d_boxSizePtr[0]) * d_boxSizePtr[0];
		deltaVec[1] -= shifty;
		distanceSq = deltaVec[0] * deltaVec[0] + deltaVec[1] * deltaVec[1];
		return sqrt(distanceSq);
		break;
		case simControlStruct::boundaryEnum::fixed:
		switch (d_simControl.geometryType) {
			case simControlStruct::geometryEnum::squareWall:
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
			case simControlStruct::geometryEnum::roundWall:
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < d_nDim; dim++) {
				delta = thisVec[dim] - otherVec[dim];
				deltaVec[dim] = delta;
				distanceSq += delta * delta;
			}
			return sqrt(distanceSq);
			break;
		}
		break;
		default: // reflect, reflectNoise, rigid and mobile
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta = thisVec[dim] - otherVec[dim];
			deltaVec[dim] = delta;
			distanceSq += delta * delta;
		}
		return sqrt(distanceSq);
		break;
	}
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

inline __device__ void getParticleVel(const long pId, const double* pVel, double* tVel) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		tVel[dim] = pVel[pId * d_nDim + dim];
	}
}

inline __device__ void getWallPos(const long wId, const double* wPos, double* tWall) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		tWall[dim] = wPos[wId * d_nDim + dim];
	}
}

inline __device__ void getPBCParticlePos(const long pId, const double* pPos, double* tPos) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		tPos[dim] = pPos[pId * d_nDim + dim];
		tPos[dim] -= floor(tPos[dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
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

inline __device__ bool extractParticleNeighborVel(const long particleId, const long nListId, const double* pVel, double* otherVel) {
	auto otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
  	if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherVel[dim] = pVel[otherId * d_nDim + dim];
    	}
    	return true;
  	}
  	return false;
}

inline __device__ bool extractVicsekNeighborAngle(const long particleId, const long nListId, const double* pAngle, double& otherAngle) {
	auto otherId = d_vicsekNeighborListPtr[particleId*d_vicsekNeighborListSize + nListId];
  	if ((particleId != otherId) && (otherId != -1)) {
		otherAngle = pAngle[otherId];
    	return true;
  	}
  	return false;
}

inline __device__ bool extractVicsekNeighborVel(const long particleId, const long nListId, const double* pVel, double* otherVel) {
	auto otherId = d_vicsekNeighborListPtr[particleId*d_vicsekNeighborListSize + nListId];
  	if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherVel[dim] = pVel[otherId * d_nDim + dim];
    	}
    	return true;
  	}
  	return false;
}

inline __device__ bool extractWallNeighborPos(const long wallId, const double* wPos, double* tWallPos) {
  	if (wallId != -1) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		tWallPos[dim] = wPos[wallId * d_nDim + dim];
    	}
    	return true;
  	}
  	return false;
}

//***************************** force and energy *****************************//
inline __device__ double calcOverlap(const double* thisVec, const double* otherVec, const double radSum) {
	return (1 - calcDistance(thisVec, otherVec) / radSum);
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
		epot = -0.5 * d_ec * (d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2) * 0.5;
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
		if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
			//printf("particleId %ld otherId %ld d_num1: %ld\n", particleId, otherId, d_num1);
			sign = 1.0;
			forceShift = d_LJfshiftPlus / radSum;
			ecut = d_LJecutPlus;
		}
		auto gradMultiple = 24 * d_ec * (2 * ratio12 + sign * ratio6) / distance - forceShift;
		auto epot = 0.5 * (4 * d_ec * (ratio12 + sign * ratio6) - ecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
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
		// interaction with neighbor particles
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
		// interaction with neighbor particles
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

inline __device__ double calcLJEnergy(const double* thisPos, const double* otherPos, const double radSum) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
		return 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
	} else {
		return 0.0;
	}
}

inline __device__ double calcWCAEnergy(const double* thisPos, const double* otherPos, const double radSum) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (WCAcut * radSum)) {
	  	return 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
	} else {
		return 0.0;
	}
}

inline __device__ double calcDoubleLJEnergy(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift = d_LJfshift / radSum;//calcDoubleLJForceShift(epsilon, radSum);
		auto epot = 0.5 * (4 * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
		// set energy scale based on particle indices
		if(particleId < d_num1) {
			if(otherId < d_num1) {
				epot *= d_eAA;
			} else {
				epot *= d_eAB;
			}
		} else {
			if(otherId >= d_num1) {
				epot *= d_eBB;
			} else {
				epot *= d_eAB;
			}
		}
		return epot;
	} else {
		return 0.0;
	}
}

inline __device__ double calcLJMinusPlusEnergy(const double* thisPos, const double* otherPos, const double radSum, const long particleId, const long otherId) {
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
		if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
			//printf("particleId %ld otherId %ld d_num1: %ld\n", particleId, otherId, d_num1);
			sign = 1.0;
			forceShift = d_LJfshiftPlus / radSum;
			ecut = d_LJecutPlus;
		}
		auto epot = 0.5 * (4 * d_ec * (ratio12 + sign * ratio6) - ecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
		return epot;
	} else {
		return 0.0;
	}
}

// AB particle-particle interaction energy
__global__ void kernelCalcParticleEnergyAB(const double* pRad, const double* pPos, const double* pVel, double* sqVel, double* pEnergyAB, long* flagAB) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto otherId = -1;
    	double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the squared velocity and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
			sqVel[particleId * d_nDim + dim] = 0;
		}
		auto thisRad = pRad[particleId];
		pEnergyAB[particleId] = 0;
		flagAB[particleId] = 0;
		// interaction with neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::doubleLJ:
					otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
					if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
						flagAB[particleId] = 1;
						pEnergyAB[particleId] += calcDoubleLJEnergy(thisPos, otherPos, radSum, particleId, otherId);
					}
					break;
					case simControlStruct::potentialEnum::LJMinusPlus:
					if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
						flagAB[particleId] = 1;
						pEnergyAB[particleId] += calcLJMinusPlusEnergy(thisPos, otherPos, radSum, particleId, otherId);
					}
					break;
					case simControlStruct::potentialEnum::LJWCA:
					otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
					if((particleId < d_num1 && otherId >= d_num1) || (particleId >= d_num1 && otherId < d_num1)) {
						flagAB[particleId] = 1;
						pEnergyAB[particleId] += calcWCAEnergy(thisPos, otherPos, radSum);
					} else {
						pEnergyAB[particleId] += calcLJEnergy(thisPos, otherPos, radSum);
					}
					break;
					default:
					break;
				}
				//if(particleId == 116 && d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId] == 109) printf("particleId %ld \t neighbor: %ld \t overlap %e \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum));
			}
		}
		if(flagAB[particleId] == 1) {
			for (long dim = 0; dim < d_nDim; dim++) {
				sqVel[particleId * d_nDim + dim] = pVel[particleId * d_nDim + dim] * pVel[particleId * d_nDim + dim];
			}
		}
  	}
}

// particle-particle kuramoto / vicsek additive interaction
__global__ void kernelCalcVicsekAdditiveAlignment(const double* pAngle, double* pAlpha) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	auto otherAngle = 0.;
		// zero out the angular acceleration and get particle positions
		pAlpha[particleId] = 0.;
		auto thisAngle = pAngle[particleId];
		// interaction with neighbor particles
		if(d_vicsekMaxNeighborListPtr[particleId] > 0) {
			for (long nListId = 0; nListId < d_vicsekMaxNeighborListPtr[particleId]; nListId++) {
				if (extractVicsekNeighborAngle(particleId, nListId, pAngle, otherAngle)) {
					auto deltaAngle = thisAngle - otherAngle;
					checkAngleMinusPIPlusPI(deltaAngle);
					pAlpha[particleId] -= d_Jvicsek * sin(deltaAngle);
				}
			}
		}
  	}
}

// particle-particle kuramoto / vicsek non-additive interaction
__global__ void kernelCalcVicsekNonAdditiveAlignment(const double* pAngle, double* pAlpha) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	auto otherAngle = 0.;
		// zero out the angular acceleration and get particle positions
		pAlpha[particleId] = 0.;
		auto thisAngle = pAngle[particleId];
		// interaction with neighbor particles
		if(d_vicsekMaxNeighborListPtr[particleId] > 0) {
			for (long nListId = 0; nListId < d_vicsekMaxNeighborListPtr[particleId]; nListId++) {
				if (extractVicsekNeighborAngle(particleId, nListId, pAngle, otherAngle)) {
					auto deltaAngle = thisAngle - otherAngle;
					checkAngleMinusPIPlusPI(deltaAngle);
					pAlpha[particleId] -= d_Jvicsek * sin(deltaAngle) / d_vicsekMaxNeighborListPtr[particleId];
				}
			}
		}
  	}
}

inline __device__ double calcWallContactInteraction(const double* thisPos, const double* wallPos, const double radSum, double* currentForce, double* wallForce) {
	double delta[MAXDIM];
	//overlap = calcOverlap(thisPos, wallPos, radSum);
	auto distance = calcDeltaAndDistance(thisPos, wallPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ew * overlap / radSum;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentForce[dim] += gradMultiple * delta[dim] / distance;
			wallForce[dim] -= gradMultiple * delta[dim] / distance;
		}
		return (0.5 * d_ew * overlap * overlap);
	}
	return 0.;
}

inline __device__ double calcWallLJInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double* wallForce) {
	double delta[MAXDIM];
	//distance = calcDistance(thisPos, otherPos);
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	//printf("distance %lf \n", distance);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (d_LJcutoff * radSum)) {
		auto forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
		auto gradMultiple = 24 * d_ew * (2 * ratio12 - ratio6) / distance - forceShift;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
			wallForce[dim] -= gradMultiple * delta[dim] / distance;
	  	}
		return 4 * d_ew * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum);
	} else {
		return 0.0;
	}
}

inline __device__ double calcWallWCAInteraction(const double* thisPos, const double* wallPos, const double radSum, double* currentForce, double* wallForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, wallPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance < (WCAcut * radSum)) {
		auto gradMultiple = 24 * d_ew * (2 * ratio12 - ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
			wallForce[dim] -= gradMultiple * delta[dim] / distance;
	  	}
	  	return d_ew * (4 * (ratio12 - ratio6) + 1);
	}
	return 0.0;
}

// particle-box contact interaction in rectangular boundary
__global__ void kernelCalcParticleSquareWallInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < (WCAcut * radSum)) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[0] - thisPos[0]) < (WCAcut * radSum)) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
		if(thisPos[1] < (WCAcut * radSum)) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < (WCAcut * radSum)) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
	}
}

// particle-sides contact interaction in 2D
__global__ void kernelCalcParticleSidesInteraction2D(const double* pRad, const double* pPos, double* pForce, double* pEnergy, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[1] < (WCAcut * radSum)) {
			wallPos[1] = 0;
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			}
		} else if((d_boxSizePtr[1] - thisPos[1]) < (WCAcut * radSum)) {
			wallPos[1] = d_boxSizePtr[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
	}
}

// particle-sides contact interaction in 3D
__global__ void kernelCalcParticleSidesInteraction3D(const double* pRad, const double* pPos, double* pForce, double* pEnergy, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[2] < (WCAcut * radSum)) {
			wallPos[2] = 0;
			wallPos[1] = thisPos[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		} else if((d_boxSizePtr[2] - thisPos[2]) < (WCAcut * radSum)) {
			wallPos[2] = d_boxSizePtr[2];
			wallPos[1] = thisPos[1];
			wallPos[0] = thisPos[0];
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			pEnergy[particleId] += calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::lennardJones:
			pEnergy[particleId] += calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			case simControlStruct::wallEnum::WCA:
			pEnergy[particleId] += calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[particleId*d_nDim]);
			break;
			default:
			break;
			}
		}
	}
}

// particle-box contact interaction in circular boundary - the center of the simulation box is assumed to be the origin
__global__ void kernelCalcParticleRoundWallInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM], wallForce[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		auto radSum = thisRad;
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
			wallForce[dim] = 0.;
		}
		auto interaction = 0.;
		// check if particle is far from the origin more than the box radius R minus the particle radius
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);
		if((d_boxRadius - thisR) < (WCAcut * radSum)) {
			wallPos[0] = d_boxRadius * cos(thisTheta);
			wallPos[1] = d_boxRadius * sin(thisTheta);
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			interaction = calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], wallForce);
			break;
			case simControlStruct::wallEnum::lennardJones:
			interaction = calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], wallForce);
			break;
			case simControlStruct::wallEnum::WCA:
			interaction = calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], wallForce);
			break;
			default:
			break;
			}
		}
		if(interaction != 0.) {
			pEnergy[particleId] += interaction;
			// transform wallForce to polar coordinates and then assign to wForce
			double radForce, thetaForce;
			cartesianToPolar(wallForce, radForce, thetaForce);
			wForce[particleId * d_nDim] = radForce;
			wForce[particleId * d_nDim + 1] = thetaForce;
		}
	}
}

// particle-box contact interaction in circular boundary - the center of the simulation box is assumed to be the origin
__global__ void kernelCalcParticleWallInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy, 
															const double* wPos, double* wForce, double* wEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		long wallId = -1;
		for (long wListId = 0; wListId < d_wallMaxNeighborListPtr[particleId]; wListId++) {
			wallId = d_wallNeighborListPtr[particleId * d_wallNeighborListSize + wListId];
			if (extractWallNeighborPos(wallId, wPos, wallPos)) {
				auto radSum = thisRad + d_wallRad;
				auto interaction = 0.;
				switch (d_simControl.wallType) {
				case simControlStruct::wallEnum::harmonic:
				interaction = calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
				pEnergy[particleId] += interaction;
				wEnergy[wallId] += interaction;
				break;
				case simControlStruct::wallEnum::lennardJones:
				interaction = calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
				pEnergy[particleId] += interaction;
				wEnergy[wallId] += interaction;
				break;
				case simControlStruct::wallEnum::WCA:
				interaction = calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
				pEnergy[particleId] += interaction;
				wEnergy[wallId] += interaction;
				break;
				default:
				break;
				}
			}
		}
	}
}

// particle-box contact interaction in circular boundary - the center of the simulation box is assumed to be the origin
__global__ void kernelCalcAllToWallParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy, 
															const double* wPos, double* wForce, double* wEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM];
		// we don't zero out the force and the energy because this function always
		// gets called after the particle-particle interaction is computed
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		for (long wallId = 0; wallId < d_numWall; wallId++) {
			getWallPos(wallId, wPos, wallPos);
			auto radSum = thisRad + d_wallRad;
			auto interaction = 0.;
			switch (d_simControl.wallType) {
			case simControlStruct::wallEnum::harmonic:
			interaction = calcWallContactInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
			pEnergy[particleId]+= interaction;
			wEnergy[wallId] += interaction;
			break;
			case simControlStruct::wallEnum::lennardJones:
			interaction = calcWallLJInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
			pEnergy[particleId]+= interaction;
			wEnergy[wallId] += interaction;
			break;
			case simControlStruct::wallEnum::WCA:
			interaction = calcWallWCAInteraction(thisPos, wallPos, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim]);
			pEnergy[particleId]+= interaction;
			wEnergy[wallId] += interaction;
			break;
			default:
			break;
			}
		}
	}
}

__global__ void kernelCalcWallAngularAcceleration(const double* wPos, const double* wForce, double* mAlpha) {
	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
	if (wallId < d_numWall) {
		double thisPos[MAXDIM];
		getWallPos(wallId, wPos, thisPos);
		auto distanceSq = 0.;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			distanceSq += thisPos[dim] * thisPos[dim];
		}
		mAlpha[wallId] = (thisPos[0] * wForce[wallId * d_nDim + 1] - thisPos[1] * wForce[wallId * d_nDim]) / distanceSq;
	}
}

__global__ void kernelCalcWallArea(const double* wPos, double* aSector) {
	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
	if (wallId < d_numWall) {
		double thisPos[MAXDIM], nextPos[MAXDIM];
		long nextId = wallId + 1;
		if(nextId > d_numWall - 1) {
			nextId = 0;
		}
		getWallPos(wallId, wPos, thisPos);
		getWallPos(nextId, wPos, nextPos);
		aSector[wallId] = thisPos[0] * nextPos[1] - nextPos[0] * thisPos[1];
	}
}

inline __device__ double calcAngle(const double* nSegment, const double* pSegment) {
	auto midSine = nSegment[0] * pSegment[1] - nSegment[1] * pSegment[0];
	auto midCosine = nSegment[0] * pSegment[0] + nSegment[1] * pSegment[1];
	return atan2(midSine, midCosine);
}

// compute segment lengths and angles from wall monomers
__global__ void kernelCalcWallShape(const double* wPos, double* wLength, double* wAngle) {
	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
	if (wallId < d_numWall) {
		double thisPos[MAXDIM], nextPos[MAXDIM], prevPos[MAXDIM];
		double delta[MAXDIM], nextSegment[MAXDIM], prevSegment[MAXDIM];
		getWallPos(wallId, wPos, thisPos);
		auto nextId = wallId + 1;
		if(nextId > d_numWall - 1) {
			nextId = 0;
		}
		auto prevId = wallId - 1;
		if(prevId < 0) {
			prevId = d_numWall - 1;
		}
		auto segmentLength = 0.;
		for (long dim = 0; dim < d_nDim; dim++) {
		delta[dim] = wPos[nextId * d_nDim + dim] - thisPos[dim];
		nextPos[dim] = thisPos[dim] + delta[dim];
		segmentLength += delta[dim] * delta[dim];
		delta[dim] = wPos[prevId * d_nDim + dim] - thisPos[dim];
		prevPos[dim] = thisPos[dim] + delta[dim];
		}
		wLength[wallId] = sqrt(segmentLength);
		getSegment(nextPos, thisPos, nextSegment);
		getSegment(thisPos, prevPos, prevSegment);
		wAngle[wallId] = calcAngle(nextSegment, prevSegment);
	}
}

inline __device__ double calcAreaForceEnergy(const double* nPos, const double* pPos, double* wForce) {
	auto deltaArea = (d_wallArea / d_wallArea0) - 1.; // area variation
	auto gradMultiple = d_ea * deltaArea / d_wallArea0;
  	wForce[0] += 0.5 * gradMultiple * (pPos[1] - nPos[1]);
  	wForce[1] += 0.5 * gradMultiple * (nPos[0] - pPos[0]);
  	return (0.5 * d_ea * deltaArea * deltaArea);
}

inline __device__ double calcPerimeterForceEnergy(const double nLength, const double nLength0, const double pLength, const double pLength0, const double* tPos, const double* nPos, const double* pPos, double* wForce) {
  	//compute length variations
  	auto pDelta = (pLength / nLength0) - 1.;
  	auto nDelta = (nLength / pLength0) - 1.;
	// compute force
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	wForce[dim] += d_el * ((nDelta * (nPos[dim] - tPos[dim]) / (nLength0 * nLength)) - (pDelta * (tPos[dim] - pPos[dim]) / (pLength0 * pLength)));
  	}
  	return (0.5 * d_el * pDelta * pDelta);
}

inline __device__ double calcBendingForceEnergy(const double* prevSegment, const double* nextSegment, 
												const double tAngleDelta, const double nAngleDelta, const double pAngleDelta, double* wForce) {
	double preNormalSegment[MAXDIM], nextNormalSegment[MAXDIM];
	// get normal segments
	getNormalVector(prevSegment, preNormalSegment);
	getNormalVector(nextSegment, nextNormalSegment);
	// compute angle variations
	auto prevVar = (tAngleDelta - pAngleDelta) / calcNormSq(prevSegment);
	auto nextVar = (tAngleDelta - nAngleDelta) / calcNormSq(nextSegment);
	// compute force
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
		wForce[dim] += d_eb * (prevVar * preNormalSegment[dim] + nextVar * nextNormalSegment[dim]);
	}
	return (0.5 * d_eb * tAngleDelta * tAngleDelta);
}

// shape interaction for mobile wall
__global__ void kernelCalcWallShapeForceEnergy(const double* wLength, const double* wAngle, const double* wPos, double* wForce, double* wEnergy) {
	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
	if (wallId < d_numWall) {
		double thisPos[MAXDIM], nextPos[MAXDIM], prevPos[MAXDIM];
		double nextSegment[MAXDIM], prevSegment[MAXDIM];
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[wallId * d_nDim + dim] = 0.;
		}
		wEnergy[wallId] = 0.;
		auto nextId = wallId + 1;
		if(nextId > d_numWall - 1) {
			nextId = 0;
		}
	  	auto prevId = wallId - 1;
		if(prevId < 0) {
			prevId = d_numWall - 1;
		}
		getWallPos(wallId, wPos, thisPos);
	  	getWallPos(nextId, wPos, nextPos);
	  	getWallPos(prevId, wPos, prevPos);
		// area force
		wEnergy[wallId] += calcAreaForceEnergy(nextPos, prevPos, &wForce[wallId*d_nDim]) / d_numWall;
		// segment force
		getSegment(nextPos, thisPos, nextSegment);
		getSegment(thisPos, prevPos, prevSegment);
	  	auto prevLength = calcNorm(prevSegment);
	  	auto nextLength = calcNorm(nextSegment);
	  	wEnergy[wallId] += calcPerimeterForceEnergy(nextLength, d_wallLength0, prevLength, d_wallLength0, thisPos, nextPos, prevPos, &wForce[wallId*d_nDim]);
		// bending force
	  	auto prevAngleDelta = wAngle[prevId] - d_wallAngle0;
	  	auto thisAngleDelta = wAngle[wallId] - d_wallAngle0;
	 	auto nextAngleDelta = wAngle[nextId] - d_wallAngle0;
		wEnergy[wallId] += calcBendingForceEnergy(prevSegment, nextSegment, thisAngleDelta, nextAngleDelta, prevAngleDelta, &wForce[wallId*d_nDim]);
	}
}

// shape interaction for plastic wall
__global__ void kernelCalcPlasticWallShapeForceEnergy(const double* wLength, const double* rLength, const double* wAngle, const double* wPos, double* wForce, double* wEnergy) {
	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
	if (wallId < d_numWall) {
		double thisPos[MAXDIM], nextPos[MAXDIM], prevPos[MAXDIM];
		double nextSegment[MAXDIM], prevSegment[MAXDIM];
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[wallId * d_nDim + dim] = 0.;
		}
		wEnergy[wallId] = 0.;
		auto nextId = wallId + 1;
		if(nextId > d_numWall - 1) {
			nextId = 0;
		}
	  	auto prevId = wallId - 1;
		if(prevId < 0) {
			prevId = d_numWall - 1;
		}
		getWallPos(wallId, wPos, thisPos);
	  	getWallPos(nextId, wPos, nextPos);
	  	getWallPos(prevId, wPos, prevPos);
		// area force
		wEnergy[wallId] += calcAreaForceEnergy(nextPos, prevPos, &wForce[wallId*d_nDim]) / d_numWall;
		// segment force
		getSegment(nextPos, thisPos, nextSegment);
		getSegment(thisPos, prevPos, prevSegment);
	  	auto prevLength = calcNorm(prevSegment);
	  	auto nextLength = calcNorm(nextSegment);
	  	wEnergy[wallId] += calcPerimeterForceEnergy(nextLength, rLength[wallId], prevLength, rLength[prevId], thisPos, nextPos, prevPos, &wForce[wallId*d_nDim]);
		// bending force
	  	auto prevAngleDelta = wAngle[prevId] - d_wallAngle0;
	  	auto thisAngleDelta = wAngle[wallId] - d_wallAngle0;
	 	auto nextAngleDelta = wAngle[nextId] - d_wallAngle0;
		wEnergy[wallId] += calcBendingForceEnergy(prevSegment, nextSegment, thisAngleDelta, nextAngleDelta, prevAngleDelta, &wForce[wallId*d_nDim]);
	}
}

inline __device__ double calcGradMultipleAndEnergy(const double* thisPos, const double* wallPos, const double radSum, double &epot) {
	double overlap, ratio, ratio6, ratio12;
	auto distance = calcDistance(thisPos, wallPos);
	switch (d_simControl.potentialType) {
		case simControlStruct::potentialEnum::harmonic:
		overlap = 1 - distance / radSum;
		if(overlap > 0) {
			epot = 0.5 * d_ec * overlap * overlap * 0.5;
			return d_ec * overlap / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::lennardJones:
		ratio = radSum / distance;
		ratio12 = pow(ratio, 12);
		ratio6 = pow(ratio, 6);
		if (distance <= (d_LJcutoff * radSum)) {
			auto forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
			epot = 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut - abs(forceShift) * (distance - d_LJcutoff * radSum));
			return 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::WCA:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance <= (WCAcut * radSum)) {
			epot = 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
			return 24 * d_ec * (2 * ratio12 - ratio6) / distance;
		} else {
			return 0;
		}
		break;
		default:
		return 0;
		break;
	}
}

// clockwise projection
inline __device__ double getProjection(const double* thisPos, const double* otherPos, const double* previousPos, const double length) {
	return ((thisPos[0] - previousPos[0]) * (otherPos[0] - previousPos[0]) + (thisPos[1] - previousPos[1]) * (otherPos[1] - previousPos[1])) / (length * length);

}

inline __device__ void getProjectionPos(const double* previousPos, const double* segment, double* projPos, const double proj) {
	auto reducedProj = max(0.0, min(1.0, proj));
	for (long dim = 0; dim < d_nDim; dim++) {
		projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
	}
}

inline __device__ double calcCross(const double* thisPos, const double* otherPos, const double* previousPos) {
	return (previousPos[0] - otherPos[0]) * (otherPos[1] - thisPos[1]) - (otherPos[0] - thisPos[0]) * (previousPos[1] - otherPos[1]);
}

inline __device__ double calcParticleSegmentInteraction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double length, const double radSum, double* thisForce, double* otherForce, double* previousForce) {
	//double segment[MAXDIM];
	auto epot = 0.0;
	auto gradMultiple = calcGradMultipleAndEnergy(thisPos, projPos, radSum, epot);
	if (gradMultiple != 0) {
		auto cross = calcCross(thisPos, otherPos, previousPos);
		auto absCross = fabs(cross);
		auto sign = cross / absCross;
		// current particle
	  	atomicAdd(&thisForce[0], gradMultiple * sign * (previousPos[1] - otherPos[1]) / length);
	  	atomicAdd(&thisForce[1], gradMultiple * sign * (otherPos[0] - previousPos[0]) / length);
		// neighbor wall monomer
	  	atomicAdd(&otherForce[0], gradMultiple * (sign * (thisPos[1] - previousPos[1]) + absCross * (previousPos[0] - otherPos[0]) / (length * length)) / length);
	  	atomicAdd(&otherForce[1], gradMultiple * (sign * (previousPos[0] - thisPos[0]) + absCross * (previousPos[1] - otherPos[1]) / (length * length)) / length);
		// previous wall monomer
	  	atomicAdd(&previousForce[0], gradMultiple * (sign * (otherPos[1] - thisPos[1]) - absCross * (previousPos[0] - otherPos[0]) / (length * length)) / length);
	  	atomicAdd(&previousForce[1], gradMultiple * (sign * (thisPos[0] - otherPos[0]) - absCross * (previousPos[1] - otherPos[1]) / (length * length)) / length);
	  	return epot;
	}
	return 0.;
}

inline __device__ double calcParticleMonomerInteraction(const double* thisPos, const double* previousPos, const double radSum, double* thisForce, double* otherForce) {
	double delta[MAXDIM];
	auto epot = 0.0;
	auto gradMultiple = calcGradMultipleAndEnergy(thisPos, previousPos, radSum, epot);
	if (gradMultiple != 0) {
		auto distance = calcDeltaAndDistance(thisPos, previousPos, delta);
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&thisForce[dim], gradMultiple * delta[dim] / distance);
			atomicAdd(&otherForce[dim], -gradMultiple * delta[dim] / distance);
		}
		return epot;
	}
	return 0.;
}

// get index of previous wall monomer
inline __device__ long getPreviousWallId(const long wallId) {
	if(wallId == 0) {
		return d_numWall - 1;
	} else {
		return wallId - 1;
  	}
}

// interaction force between particles and wall monomers + segments
__global__ void kernelCalcSmoothWallInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy,
												const double* wPos, double* wForce, double* wEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto interaction = 0.;
		double thisPos[MAXDIM], wallPos[MAXDIM], prevPos[MAXDIM], secondPrevPos[MAXDIM];
		double projPos[MAXDIM], segment[MAXDIM], prevSegment[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		// loop through wall monomer in wallNeighborList
		long wallId = -1;
		for (long wListId = 0; wListId < d_wallMaxNeighborListPtr[particleId]; wListId++) {
			wallId = d_wallNeighborListPtr[particleId * d_wallNeighborListSize + wListId];
			if (extractWallNeighborPos(wallId, wPos, wallPos)) {
				auto radSum = thisRad + d_wallRad;
				// compute projection of vertexId on segment between otherId and previousId
				auto prevId = getPreviousWallId(wallId);
				getWallPos(prevId, wPos, prevPos);
				getDelta(wallPos, prevPos, segment);
				auto length = calcNorm(segment);
				auto proj = getProjection(thisPos, wallPos, prevPos, length);
				// compare projection with previous vertex - concave particles can interact with two consecutive segments
				auto secondPrevId = getPreviousWallId(prevId);
				getWallPos(secondPrevId, wPos, secondPrevPos);
				getDelta(secondPrevPos, prevPos, prevSegment);
				auto prevLength = calcNorm(prevSegment);
				auto prevProj = getProjection(thisPos, prevPos, secondPrevPos, prevLength);
				// check if the interaction is vertex-segment
				bool isPrevMonomerSegment = false;
				if(proj >= 0 && proj < 1) {
					getProjectionPos(prevPos, segment, projPos, proj);
					interaction = calcParticleSegmentInteraction(thisPos, projPos, wallPos, prevPos, length, radSum, &pForce[particleId*d_nDim], &wForce[wallId*d_nDim], &wForce[prevId*d_nDim]);
					atomicAdd(&pEnergy[particleId], interaction);
					atomicAdd(&wEnergy[wallId], interaction); // energy should be split and assigned to wallId and prevId
					if(prevProj >= 0 && prevProj < 1) {
						// wall is concave - subtract excessive particle-monomer interaction with shared monomer
						isPrevMonomerSegment = true;
						interaction = calcParticleMonomerInteraction(prevPos, thisPos, radSum, &pForce[particleId*d_nDim], &wForce[prevId*d_nDim]);
						atomicAdd(&pEnergy[particleId], -interaction);
						atomicAdd(&wEnergy[prevId], -interaction);
					}
				} else if(proj < 0) {
					if(prevProj >= 1 && isPrevMonomerSegment == false) {
					// wall is convex - add particle-monomer interaction
					interaction = calcParticleMonomerInteraction(thisPos, prevPos, radSum, &pForce[particleId*d_nDim], &wForce[prevId*d_nDim]);
					atomicAdd(&pEnergy[particleId], interaction);
					atomicAdd(&pEnergy[prevId], interaction);
					}
				}
			}
		}
  	}
}

__global__ void kernelCheckParticleInsideRoundWall(double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		// check if particle is outside round box
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);
		if(thisR > d_boxRadius) {
			//auto radialDistance = thisR - d_boxRadius;
			//printf("particle %ld is outside of the box at distance %lf\n", particleId, radialDistance);
			thisR = d_boxRadius;
			polarToCartesian(thisPos, thisR, thisTheta);
			for (long dim = 0; dim < d_nDim; dim++) {
				pPos[particleId * d_nDim + dim] = thisPos[dim];
			}
		}
	}
}

__global__ void kernelReflectParticleFixedWall(const double* pRad, const double* pPos, double* pVel, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], thisVel[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		// save previous velocity to compute momentum exchange across wall
		getParticleVel(particleId, pVel, thisVel);
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		bool isWallx = false;
		if(thisPos[0] < thisRad) {
			isWallx = true;
			pVel[particleId * d_nDim] = -pVel[particleId * d_nDim];
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			isWallx = true;
			pVel[particleId * d_nDim] = -pVel[particleId * d_nDim];
		}
		bool isWally = false;
		if(thisPos[1] < thisRad) {
			isWally = true;
			pVel[particleId * d_nDim + 1] = -pVel[particleId * d_nDim + 1];
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			isWally = true;
			pVel[particleId * d_nDim + 1] = -pVel[particleId * d_nDim + 1];
		}
		if(isWallx || isWally) {
		// compute momentum exchange and assign it to wForce
			for (long dim = 0; dim < d_nDim; dim++) {
				wForce[particleId * d_nDim + dim] = (pVel[particleId * d_nDim + dim] - thisVel[dim]) / d_dt;
			}
		}
	}
}

__global__ void kernelReflectParticleFixedSides2D(const double* pRad, const double* pPos, double* pVel, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], thisVel[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		// save previous velocity to compute momentum exchange across wall
		getParticleVel(particleId, pVel, thisVel);
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		bool isWall = false;
		if(thisPos[1] < thisRad) {
			isWall = true;
			pVel[particleId * d_nDim + 1] = -pVel[particleId * d_nDim + 1];
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			isWall = true;
			pVel[particleId * d_nDim + 1] = -pVel[particleId * d_nDim + 1];
		}
		if(isWall) {
		// compute momentum exchange and assign it to wForce
			for (long dim = 0; dim < d_nDim; dim++) {
				wForce[particleId * d_nDim + dim] = (pVel[particleId * d_nDim + dim] - thisVel[dim]) / d_dt;
			}
		}
	}
}

__global__ void kernelReflectParticleFixedSides3D(const double* pRad, const double* pPos, double* pVel, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], thisVel[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		// save previous velocity to compute momentum exchange across wall
		getParticleVel(particleId, pVel, thisVel);
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
		}
		bool isWall = false;
		if(thisPos[2] < thisRad) {
			isWall = true;
			pVel[particleId * d_nDim + 2] = -pVel[particleId * d_nDim + 2];
		} else if((d_boxSizePtr[2] - thisPos[2]) < thisRad) {
			isWall = true;
			pVel[particleId * d_nDim + 2] = -pVel[particleId * d_nDim + 2];
		}
		if(isWall) {
		// compute momentum exchange and assign it to wForce
			for (long dim = 0; dim < d_nDim; dim++) {
				wForce[particleId * d_nDim + dim] = (pVel[particleId * d_nDim + dim] - thisVel[dim]) / d_dt;
			}
		}
	}
}

__global__ void kernelReflectParticleRoundWall(const double* pRad, const double* pPos, double* pVel, double* pAngle, double* wForce) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], thisVel[MAXDIM], wallForce[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		// check if particle is far from the origin more than the box radius R minus the particle radius
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);
		// save previous velocity to compute momentum exchange across wall
		getParticleVel(particleId, pVel, thisVel);
		for (long dim = 0; dim < d_nDim; dim++) {
			wForce[particleId*d_nDim + dim] = 0.;
			wallForce[dim] = 0.;
		}
		if((d_boxRadius - thisR) < thisRad) { // should replace thisRad with zero?
			auto vDotn = pVel[particleId * d_nDim] * cos(thisTheta) + pVel[particleId * d_nDim + 1] * sin(thisTheta);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(thisTheta);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(thisTheta);
			auto velAngle = atan2(pVel[particleId * d_nDim + 1], pVel[particleId * d_nDim]);
			switch (d_simControl.particleType) {
				case simControlStruct::particleEnum::active:
				case simControlStruct::particleEnum::vicsek:
				pAngle[particleId] = velAngle;
				break;
				default:
				break;
			}
			// compute momentum exchange and assign it to wForce
			for (long dim = 0; dim < d_nDim; dim++) {
				wallForce[dim] = (pVel[particleId * d_nDim + dim] - thisVel[dim]) / d_dt;
			}
			// transform wallForce to polar coordinates and then assign to wForce
			double radForce, thetaForce;
			cartesianToPolar(wallForce, radForce, thetaForce);
			wForce[particleId * d_nDim] = radForce;
			wForce[particleId * d_nDim + 1] = thetaForce;
		}
	}
}

__global__ void kernelReflectParticleFixedWallWithNoise(const double* pRad, const double* pPos, const double* randAngle, double* pVel) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		if(thisPos[0] < thisRad) {
			// add Gaussian noise to the angle of reflection
			auto reflectAngle = randAngle[particleId];
			checkAngleMinusPIPlusPI(reflectAngle);
			auto vDotn = pVel[particleId * d_nDim] * cos(reflectAngle) + pVel[particleId * d_nDim + 1] * sin(reflectAngle);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(reflectAngle);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(reflectAngle);
		} else if((d_boxSizePtr[0] - thisPos[0]) < thisRad) {
			// add Gaussian noise to the angle of reflection
			auto reflectAngle = -PI + randAngle[particleId];
			checkAngleMinusPIPlusPI(reflectAngle);
			auto vDotn = pVel[particleId * d_nDim] * cos(reflectAngle) + pVel[particleId * d_nDim + 1] * sin(reflectAngle);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(reflectAngle);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(reflectAngle);
		}
		if(thisPos[1] < thisRad) {
			// add Gaussian noise to the angle of reflection
			auto reflectAngle = -0.5 * PI + randAngle[particleId];
			checkAngleMinusPIPlusPI(reflectAngle);
			auto vDotn = pVel[particleId * d_nDim] * cos(reflectAngle) + pVel[particleId * d_nDim + 1] * sin(reflectAngle);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(reflectAngle);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(reflectAngle);
		} else if((d_boxSizePtr[1] - thisPos[1]) < thisRad) {
			// add Gaussian noise to the angle of reflection
			auto reflectAngle = 0.5 * PI + randAngle[particleId];
			checkAngleMinusPIPlusPI(reflectAngle);
			auto vDotn = pVel[particleId * d_nDim] * cos(reflectAngle) + pVel[particleId * d_nDim + 1] * sin(reflectAngle);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(reflectAngle);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(reflectAngle);
		}
	}
}

__global__ void kernelReflectParticleRoundWallWithNoise(const double* pRad, const double* pPos, const double* randAngle, double* pVel) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::none:
			thisRad = 0.;
			break;
			default:
			break;
		}
		// check if particle is far from the origin more than the box radius R minus the particle radius
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);
		if((d_boxRadius - thisR) < thisRad) {
			// add Gaussian noise to the angle of reflection
			auto reflectAngle = thisTheta + randAngle[particleId];
			auto vDotn = pVel[particleId * d_nDim] * cos(reflectAngle) + pVel[particleId * d_nDim + 1] * sin(reflectAngle);
			pVel[particleId * d_nDim] = pVel[particleId * d_nDim] - 2 * vDotn * cos(reflectAngle);
			pVel[particleId * d_nDim + 1] = pVel[particleId * d_nDim + 1] - 2 * vDotn * sin(reflectAngle);
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

__global__ void kernelCalcStressTensor(const double* pRad, const double* pPos, const double* pVel, double* pStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], force[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		// thermal stress
		//for (long dim = 0; dim < d_nDim; dim++) {
		//	pStress[dim * d_nDim + dim] += pVel[particleId * d_nDim + dim] * pVel[particleId * d_nDim + dim];
		//}
		// stress between neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
			if(extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				auto gradMultiple = calcGradMultiple(particleId, otherId, thisPos, otherPos, radSum);
				if(gradMultiple != 0.0) {
					auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
					for (long dim = 0; dim < d_nDim; dim++) {
						force[dim] = gradMultiple * delta[dim] / distance;
					}
					//diagonal terms
					pStress[0] += delta[0] * force[0];
					pStress[3] += delta[1] * force[1];
					// cross terms
					pStress[1] += delta[0] * force[1];
					pStress[2] += delta[1] * force[0];
				}
			}
		}
	}
}

__global__ void kernelCalcActiveStress(const double* pAngle, const double* pVel, double* pStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		// thermal stress
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
			pStress[dim * d_nDim + dim] += 0.5 * (d_driving * d_taup) * ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId])) * pVel[particleId * d_nDim + dim];
		}
	}
}

inline __device__ void calcWallStress(const double* thisPos, const double* wallPos, const double radSum, double* force) {
  	auto distanceSq = 0.0;
	auto gradMultiple = 0.0;
	double ratio, ratio6, ratio12, forceShift;
	double delta[MAXDIM];
	for (long dim = 0; dim < d_nDim; dim++) {
		force[dim] = 0.0;
		delta[dim] = thisPos[dim] - wallPos[dim];
		distanceSq += delta[dim] * delta[dim];
	}
	auto distance = sqrt(distanceSq);
	auto overlap = 1 - distance / radSum;
	switch (d_simControl.potentialType) {
		case simControlStruct::potentialEnum::harmonic:
		overlap = 1 - distance / radSum;
		if(overlap > 0) {
			gradMultiple = d_ew * overlap / radSum;
		}
		break;
		case simControlStruct::potentialEnum::adhesive:
		overlap = 1 - distance / radSum;
		if (distance < (1 + d_l1) * radSum) {
			gradMultiple = d_ew * overlap / radSum;
		} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			gradMultiple = -(d_ew * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		}
		break;
		case simControlStruct::potentialEnum::WCA:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (WCAcut * radSum)) {
			gradMultiple = 24 * d_ew * (2 * ratio12 - ratio6) / distance;
		}
		break;
		default:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance < (d_LJcutoff * radSum)) {
			forceShift =  d_LJfshift / radSum;//calcLJForceShift(radSum);
			gradMultiple = 24 * d_ew * (2 * ratio12 - ratio6) / distance - forceShift;
		}
		break;
	}
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
		force[dim] = gradMultiple * delta[dim] / distance;
	}
}

__global__ void kernelCalcWallStress(const double* pRad, const double* pPos, double* wallStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM], force[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    	auto radSum = pRad[particleId];
		auto range = 0.0;
		wallStress[particleId] = 0;
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::harmonic:
			range = radSum;
			break;
			case simControlStruct::potentialEnum::adhesive:
			range = (1 + d_l2) * radSum;
			break;
			case simControlStruct::potentialEnum::WCA:
			range = WCAcut * radSum;
			break;
			default:
			range = d_LJcutoff * radSum;
			break;
		}
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[0] < range) {
			wallPos[0] = 0;
			wallPos[1] = thisPos[1];
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[0];
		} else if((d_boxSizePtr[0] - thisPos[0]) < range) {
			wallPos[0] = d_boxSizePtr[0];
			wallPos[1] = thisPos[1];
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[0];
		}
		if(thisPos[1] < range) {
			wallPos[0] = thisPos[0];
			wallPos[1] = 0;
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[1];
		} else if((d_boxSizePtr[1] - thisPos[1]) < range) {
			wallPos[0] = thisPos[0];
			wallPos[1] = d_boxSizePtr[1];
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[1];
		}
	}
}

__global__ void kernelCalcSides2DStress(const double* pRad, const double* pPos, double* wallStress) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], wallPos[MAXDIM], force[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    	auto radSum = pRad[particleId];
		auto range = 0.0;
		wallStress[particleId] = 0;
		switch (d_simControl.potentialType) {
			case simControlStruct::potentialEnum::harmonic:
			range = radSum;
			break;
			case simControlStruct::potentialEnum::adhesive:
			range = (1 + d_l2) * radSum;
			break;
			case simControlStruct::potentialEnum::WCA:
			range = WCAcut * radSum;
			break;
			default:
			range = d_LJcutoff * radSum;
			break;
		}
		// check if particle is close to the wall at a distance less than its radius
		if(thisPos[1] < range) {
			wallPos[0] = thisPos[0];
			wallPos[1] = 0;
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[1];
		} else if((d_boxSizePtr[1] - thisPos[1]) < range) {
			wallPos[0] = thisPos[0];
			wallPos[1] = d_boxSizePtr[1];
			calcWallStress(thisPos, wallPos, radSum, force);
			wallStress[particleId] += force[1];
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
__global__ void kernelCalcWallForce(const double* pRad, const double* pPosPBC, const double range, double* wallForce, long* wallCount) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		long otherId;
		auto midHeight = d_boxSizePtr[1] * 0.5;
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPosPBC, thisPos);
		wallForce[particleId] = 0.;
		wallCount[particleId] = 0;
		if(d_partMaxNeighborListPtr[particleId] > 0) {
			auto thisRad = pRad[particleId];
			auto thisHeight = thisPos[1];// - d_boxSizePtr[1] * floor(thisPos[1] / d_boxSizePtr[1]);
			//thisDistance = thisPos[1] - midHeight;
			//if(thisDistance < 0) {
			if(thisHeight < midHeight && thisHeight > (midHeight - range)) {
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

// particle-particle interaction across fictitious wall at half height in 2D and width less then box width
__global__ void kernelCalcCenterWallForce(const double* pRad, const double* pPosPBC, const double range, const double width, double* wallForce, long* wallCount) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		long otherId;
		auto midHeight = d_boxSizePtr[1] * 0.5;
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPosPBC, thisPos);
		wallForce[particleId] = 0.;
		wallCount[particleId] = 0;
		auto lowBound = (d_boxSizePtr[0] - width) * 0.5;
		auto highBound = (d_boxSizePtr[0] + width) * 0.5;
		if(thisPos[0] > lowBound && thisPos[0] < highBound) {
			if(d_partMaxNeighborListPtr[particleId] > 0) {
				auto thisRad = pRad[particleId];
				auto thisHeight = thisPos[1];
				if(thisHeight < midHeight && thisHeight > (midHeight - range)) {
					for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
						if (extractParticleNeighbor(particleId, nListId, pPosPBC, pRad, otherPos, otherRad)) {
							if(otherPos[0] > lowBound && otherPos[0] < highBound) {
								auto otherHeight = otherPos[1];
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
								}
							}
						}
					}
				}
			}
		}
  	}
}

// particle-particle interaction across fictitious wall at half height in 2D
__global__ void kernelAddWallActiveForce(const double* pAngle, double* wallForce, long* wallCount) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		// if the interaction of particleId has already been counted
		if(wallCount[particleId] > 0) {
			wallForce[particleId] += d_driving * sin(pAngle[particleId]);
		}
  	}
}

__global__ void kernelCalcColumnWork(const double* pRad, const double* pPos, const double* pVel, const double width, double &workIn, double &workOut) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double otherRad, thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], force[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		auto work = 0.0;
		// thermal stress
		for(long dim = 0; dim < d_nDim; dim++) {
			work += pVel[particleId * d_nDim + dim] * pVel[particleId * d_nDim + dim];
		}
		// stress between neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
			if(extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				//if(otherPos[0] > lowBound && otherPos[1] < highBound) {
				auto radSum = thisRad + otherRad;
				auto gradMultiple = calcGradMultiple(particleId, otherId, thisPos, otherPos, radSum);
				if(gradMultiple != 0.0) {
					auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
					for (long dim = 0; dim < d_nDim; dim++) {
						force[dim] = gradMultiple * delta[dim] / distance;
						work += 0.5 * (force[dim] * delta[dim]);
					}
				}
			}
		}
		auto lowBound = (d_boxSizePtr[0] - width) * 0.5;
		auto highBound = (d_boxSizePtr[0] + width) * 0.5;
		if(thisPos[0] > lowBound && thisPos[1] < highBound) {
			workIn += work;
		} else {
			workOut += work;
		}
	}
}

__global__ void kernelCalcColumnActiveWork(const double* pPos, const double* pAngle, const double* pVel, const double width, double &aWorkIn, double aWorkOut) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto activeWork = 0.0;
		for (long dim = 0; dim < d_nDim; dim++) {
			activeWork += ((1 - dim) * cos(pAngle[particleId]) + dim * sin(pAngle[particleId])) * pVel[particleId * d_nDim + dim];
		}
		auto lowBound = (d_boxSizePtr[0] - width) * 0.5;
		auto highBound = (d_boxSizePtr[0] + width) * 0.5;
		if(thisPos[0] > lowBound && thisPos[1] < highBound) {
			aWorkIn += activeWork;
		} else {
			aWorkOut += activeWork;
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

__global__ void kernelCalcVicsekNeighborList(const double* pPos, const double* pRad, const double Rvicsek) {
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
				isNeighbor = (calcDistance(thisPos, otherPos) < (Rvicsek * radSum));
				if (addedNeighbor < d_vicsekNeighborListSize) {
					d_vicsekNeighborListPtr[particleId * d_vicsekNeighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					//if(isNeighbor == true) printf("particleId %ld \t otherId: %ld \t isNeighbor: %i \n", particleId, otherId, isNeighbor);
				}
				addedNeighbor += isNeighbor;
			}
		}
		d_vicsekMaxNeighborListPtr[particleId] = addedNeighbor;
  	}
}

// list of neighboring monomers on wall (ring polymer)
__global__ void kernelCalcParticleWallNeighborList(const double* pPos, const double* pRad, const double* wPos, const double cutDistance) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto addedNeighbor = 0;
		double thisPos[MAXDIM], wallPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];
		// get polar coordinates
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);

		for (long wallId = 0; wallId < d_numWall; wallId++) {
			getWallPos(wallId, wPos, wallPos);
			bool isNeighbor = false;
			auto radSum = thisRad + d_wallRad;
			isNeighbor = (-calcOverlap(thisPos, wallPos, radSum) < cutDistance);// cutDistance should be greater than radSum
			//isNeighbor = (calcDistance(thisPos, otherPos) < (cutDistance * radSum));
			if (addedNeighbor < d_wallNeighborListSize) {
				d_wallNeighborListPtr[particleId * d_wallNeighborListSize + addedNeighbor] = wallId*isNeighbor -1*(!isNeighbor);
				//if(isNeighbor == true) printf("particleId %ld \t otherId: %ld \t isNeighbor: %i \n", particleId, otherId, isNeighbor);
			}
			addedNeighbor += isNeighbor;
		}
		d_wallMaxNeighborListPtr[particleId] = addedNeighbor;
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

__global__ void kernelAssignWallLabel(const double* pPos, const double* pRad, long* wallLabel, long direction_) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		wallLabel[particleId] = 0;
		getPBCParticlePos(particleId, pPos, thisPos);
		double distanceCheck = 4 * pRad[particleId];
		long checkIdx = -1;
		if(direction_ == 1) {
			checkIdx = 0; // compression along x axis
		} else if(direction_ == 0) {
			checkIdx = 1; // compression along y axis
		}
		if(checkIdx != -1) {
			if(thisPos[checkIdx] < distanceCheck) {
				wallLabel[particleId] = 1;
			} else if((d_boxSizePtr[checkIdx] - thisPos[checkIdx]) < distanceCheck) {
				wallLabel[particleId] = 1;
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
		flag[particleId] = 0;
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

// particle-particle vicsek interaction
__global__ void kernelCalcVicsekAngleAlignment(double* pAngle) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	auto otherAngle = 0.;
		// zero out the angular acceleration and get particle positions
		auto thisAngle = 0.;
		// interaction with neighbor particles
		if(d_vicsekMaxNeighborListPtr[particleId] > 0) {
			for (long nListId = 0; nListId < d_vicsekMaxNeighborListPtr[particleId]; nListId++) {
				if (extractVicsekNeighborAngle(particleId, nListId, pAngle, otherAngle)) {
					thisAngle += otherAngle;
				}
			}
			thisAngle /= d_vicsekMaxNeighborListPtr[particleId];
		}
		pAngle[particleId] = thisAngle;
  	}
}

// vicsek average unit velocity
__global__ void kernelCalcUnitVelocity(const double* pVel, double* unitVel) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		// compute unit vector magnitude
		auto velAngle = atan2(pVel[particleId * d_nDim + 1], pVel[particleId * d_nDim]);
		unitVel[particleId * d_nDim] = cos(velAngle);
		unitVel[particleId * d_nDim + 1] = sin(velAngle);
  	}
}

// vorticity parameter
__global__ void kernelCalcVortexParameters(const double* pPos, const double* pVel, double* vortexParam) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		double thisPos[MAXDIM];
		// get driving angle and position angle from polar coordinates
		auto velAngle = atan2(pVel[particleId * d_nDim + 1], pVel[particleId * d_nDim]);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
		double thisR, thisTheta;
		cartesianToPolar(thisPos, thisR, thisTheta);
		vortexParam[particleId] = sin(thisTheta - velAngle);
	}
}

// vicsek velocity correlation
__global__ void kernelCalcVicsekVelocityCorrelation(const double* pVel, double* velCorr) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	double thisVel[MAXDIM], otherVel[MAXDIM];
		auto thisVelSquared =  0.;
		// zero out squared velocity components
		for (long dim = 0; dim < d_nDim; dim++) {
			thisVel[dim] = pVel[particleId * d_nDim + dim];
			thisVelSquared += thisVel[dim] * thisVel[dim];
		}
		velCorr[particleId] = 0.;
		// correlation with vicsek neighbor particles
		if(d_vicsekMaxNeighborListPtr[particleId] > 0) {
			for (long nListId = 0; nListId < d_vicsekMaxNeighborListPtr[particleId]; nListId++) {
				if (extractVicsekNeighborVel(particleId, nListId, pVel, otherVel)) {
					for (long dim = 0; dim < d_nDim; dim++) {
						velCorr[particleId] += thisVel[dim] * otherVel[dim];
					}
				}
			}
			velCorr[particleId] /= (thisVelSquared * d_vicsekMaxNeighborListPtr[particleId]);
		}
  	}
}

// neighbor velocity correlation
__global__ void kernelCalcNeighborVelocityCorrelation(const double* pVel, double* velCorr) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	double thisVel[MAXDIM], otherVel[MAXDIM];
		auto thisVelSquared =  0.;
		// zero out squared velocity components
		for (long dim = 0; dim < d_nDim; dim++) {
			thisVel[dim] = pVel[particleId * d_nDim + dim];
			thisVelSquared += pVel[particleId * d_nDim + dim] * pVel[particleId * d_nDim + dim];
		}
		velCorr[particleId] = 0.;
		// correlation with neighbor particles
		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
			if (extractParticleNeighborVel(particleId, nListId, pVel, otherVel)) {
				for (long dim = 0; dim < d_nDim; dim++) {
					velCorr[particleId] += thisVel[dim] * otherVel[dim];
				}
			}
		}
		if(d_partMaxNeighborListPtr[particleId] != 0) {
			velCorr[particleId] /= (thisVelSquared * d_partMaxNeighborListPtr[particleId]);
		}
  	}
}

// neighbor velocity correlation
__global__ void kernelCalcParticleAngularMomentum(const double* pPos, const double* pVel, double* angMom) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		angMom[particleId] = pPos[particleId * d_nDim] * pVel[particleId * d_nDim + 1] - pPos[particleId * d_nDim + 1] * pVel[particleId * d_nDim];
  	}
}

//******************************** integrators *******************************//
__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] += timeStep * pVel[particleId * d_nDim + dim];
		}
  	}
}

__global__ void kernelSetPBC(double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] -= floor(pPos[particleId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
		}
	}
}

__global__ void kernelCheckParticlePBC(double* pPosPBC, const double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			pPosPBC[particleId * d_nDim + dim] = pPos[particleId * d_nDim + dim] - floor(pPos[particleId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
		}
	}
}

__global__ void kernelUpdateParticleVel(double* pVel, const double* pForce, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] += timeStep * pForce[particleId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateParticleAngle(double* pAngle, const double* pOmega, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		pAngle[particleId] += timeStep * pOmega[particleId];
		pAngle[particleId] = pAngle[particleId] + PI;
		pAngle[particleId] = pAngle[particleId] - 2.0 * PI * floor(pAngle[particleId] / (2.0 * PI));
		pAngle[particleId] = pAngle[particleId] - PI;
  	}
}

__global__ void kernelUpdateParticleOmega(double* pOmega, const double* pAlpha, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		pOmega[particleId] += timeStep * pAlpha[particleId];
  	}
}

__global__ void kernelSumParticleVelocity(double* pVel, double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			velSum[dim] += pVel[particleId * d_nDim + dim];
		}
	}
}

__global__ void kernelSubtractParticleDrift(double* pVel, double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= velSum[dim]/d_numParticles;
		}
	}
}

__global__ void kernelSubsetSumParticleVelocity(double* pVel, double* velSum, long firstId) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles && particleId > firstId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			velSum[dim] += pVel[particleId * d_nDim + dim];
		}
	}
}

__global__ void kernelSubsetSubtractParticleDrift(double* pVel, double* velSum, long firstId) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles && particleId > firstId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= velSum[dim]/d_numParticles;
		}
	}
}

__global__ void kernelUpdateWallPos(double* wPos, const double* wVel, const double timeStep) {
  	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (wallId < d_numWall) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			wPos[wallId * d_nDim + dim] += timeStep * wVel[wallId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateWallVel(double* wVel, const double* wForce, const double timeStep) {
  	long wallId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (wallId < d_numWall) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			wVel[wallId * d_nDim + dim] += timeStep * wForce[wallId * d_nDim + dim];
		}
  	}
}


#endif /* DPM2DKERNEL_CUH_ */
