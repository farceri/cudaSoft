//
// Author: Francesco Arceri
// Date:   10-01-2021
//
// HEADER FILE FOR SP2D CLASS

#ifndef SP2D_H
#define SP2D_H

#include "defs.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using std::vector;
using std::string;

struct simControlStruct {
  enum class geometryEnum {normal, leesEdwards, fixedBox, fixedSides2D, fixedSides3D} geometryType;
  enum class interactionEnum {neighbor, allToAll} interactionType;
  enum class potentialEnum {harmonic, lennardJones, Mie, WCA, adhesive, doubleLJ} potentialType;
  enum class boxEnum {harmonic, WCA} boxType;
  enum class gravityEnum {on, off} gravityType;
};

// pointer-to-member function call macro
#define CALL_MEMBER_FN(object, ptrToMember) ((object).*(ptrToMember))

class SP2D;
class FIRE;
class SimInterface;
//typedef void (SP2D::*sp2d)(void);

class SP2D
{
public:

  // constructor and deconstructor
  SP2D(long nParticles, long dim);
  ~SP2D();

  // Simulator
  FIRE * fire_;
  SimInterface * sim_;

  simControlStruct simControl;

  // variables for CUDA runtime details
  long dimGrid, dimBlock;

  // sp packing constants
  long nDim;
  long numParticles;

  thrust::device_vector<double> d_boxSize;

  // time step
  double dt;
  // dimensional factor
  double rho0;
  // energy scale
  double ec;
  // adhesion constants
  double l1, l2;
  // Lennard-Jones constants
  double LJcutoff, LJecut;
  // double Lennard-Jones constants
  double eAA, eAB, eBB;
  // Mie constants
  double nPower, mPower;
  double mieConstant, Miecut;
  // Lees-Edwards shift
  double LEshift;
  // Gravity
  double gravity, ew;
  // Fluid flow
  double flowSpeed, flowDecay;
  double flowViscosity, flowHeight;
  // neighbor update variables
  double cutoff, cutDistance;
  long updateCount;

  // dynamical particle variables
  thrust::device_vector<double> d_particlePos;
  thrust::device_vector<double> d_particleRad;
  thrust::device_vector<double> d_particleVel;
  thrust::device_vector<double> d_particleLastVel;
  thrust::device_vector<double> d_particleForce;
  thrust::device_vector<double> d_particleEnergy;
  thrust::device_vector<double> d_particleAngle;
  thrust::device_vector<double> d_stress;
  thrust::device_vector<double> d_wallForce;
  thrust::device_vector<long> d_wallCount;
  // hydrodynamical variables
  thrust::device_vector<double> d_flowVel;
  thrust::device_vector<double> d_surfaceHeight;

  // correlation variables
  thrust::device_vector<double> d_particleInitPos;
  thrust::device_vector<double> d_particleLastPos;
	thrust::device_vector<double> d_particleDelta;
  thrust::device_vector<double> d_particleDisp;

  // contact list
  thrust::device_vector<long> d_numContacts;
  thrust::device_vector<long> d_contactList;
  thrust::device_vector<double> d_contactVectorList;
  long maxContacts;
  long contactLimit;

	// neighbor list
  thrust::device_vector<long> d_partNeighborList;
  thrust::device_vector<long> d_partMaxNeighborList;
  long partMaxNeighbors;
	long partNeighborListSize;
  long neighborLimit;

  double checkGPUMemory();

  void initParticleVariables(long numParticles_);

  void initParticleDeltaVariables(long numParticles_);

  void initContacts(long numParticles_);

  void initParticleNeighbors(long numParticles_);

  //setters and getters
  void syncSimControlToDevice();
  void syncSimControlFromDevice();
  bool testSimControlSync();

  void setGeometryType(simControlStruct::geometryEnum geometryType_);
	simControlStruct::geometryEnum getGeometryType();

  void setInteractionType(simControlStruct::interactionEnum interactionType_);
	simControlStruct::interactionEnum getInteractionType();

  void setPotentialType(simControlStruct::potentialEnum potentialType_);
	simControlStruct::potentialEnum getPotentialType();

  void setBoxType(simControlStruct::boxEnum boxType_);
	simControlStruct::boxEnum getBoxType();

  void setGravityType(simControlStruct::gravityEnum gravityType_);
	simControlStruct::gravityEnum getGravityType();

  void setLEshift(double LEshift_);
  double getLEshift();

  void applyLEShear(double LEshift_);

  void applyExtension(double shifty_);

  void applyUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, long direction_);

  void applyCenteredUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, long direction_);

  void applyBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, double shiftx_);

  void applyCenteredBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double shifty_, double shiftx_);
  
  void setDimBlock(long dimBlock_);
  long getDimBlock();

  void setNDim(long nDim_);
  long getNDim();

  void setNumParticles(long numParticles_);
	long getNumParticles();

  void setParticleLengthScale();

  void setLengthScaleToOne();

  void setBoxSize(thrust::host_vector<double> &boxSize_);
  thrust::host_vector<double> getBoxSize();

  void setParticleRadii(thrust::host_vector<double> &particleRad_);
  thrust::host_vector<double> getParticleRadii();

  double getMeanParticleSigma();

  double getMinParticleSigma();

  double getMaxParticleSigma();

  void setPBC();

  void setParticlePositions(thrust::host_vector<double> &particlePos_);
  void setPBCParticlePositions(thrust::host_vector<double> &particlePos_);
  thrust::host_vector<double> getParticlePositions();
  thrust::host_vector<double> getPBCParticlePositions();

  thrust::host_vector<double> getParticleDeltas();

  void resetLastPositions();

  void setInitialPositions();

  thrust::host_vector<double> getLastPositions();

  void resetLastVelocities();

  void setParticleVelocities(thrust::host_vector<double> &particleVel_);
  thrust::host_vector<double> getParticleVelocities();

  void setParticleForces(thrust::host_vector<double> &particleForce_);
  thrust::host_vector<double> getParticleForces();

  thrust::host_vector<double> getParticleEnergies();

  void setParticleAngles(thrust::host_vector<double> &particleAngle_);
  thrust::host_vector<double> getParticleAngles();

  thrust::host_vector<double> getPerParticleStressTensor();

  thrust::host_vector<double> getStressTensor();

  thrust::host_vector<long> getContacts();

  void printContacts();

  double getParticlePhi();

  double getParticleMSD();

  double getParticleMaxDisplacement();

  double setDisplacementCutoff(double cutoff_);

  void resetUpdateCount();

  long getUpdateCount();

  void checkParticleMaxDisplacement();

  void checkParticleMaxDisplacement2();

  double getSoftWaveNumber();

  double getParticleISF(double waveNumber_);

  // initialization functions
  void setPolyRandomParticles(double phi0, double polyDispersity);

  void setScaledPolyRandomParticles(double phi0, double polyDispersity, double lx);

  void setScaledMonoRandomParticles(double phi0, double lx);

  void pressureScaleParticles(double pscale);

  void scaleParticles(double scale);

  void scaleParticlePacking();

  void scaleParticleVelocity(double scale);

  void computeParticleAngleFromVel();

  // force and energy
  void setEnergyCostant(double ec_);

  double setTimeStep(double dt_);

  void setAdhesionParams(double l1_, double l2_);

  void setLJcutoff(double LJcutoff_);

  void setDoubleLJconstants(double LJcutoff_, double eAA_, double eAB_, double eBB_);

  void setMieParams(double LJcutoff_, double nPower_, double mPower_);

  void setBoxEnergyScale(double ew_);

  void setGravity(double gravity_, double ew_);

  void setFluidFlow(double speed_, double viscosity_);

  void calcSurfaceHeight();

  double getSurfaceHeight();

  void calcFlowVelocity();

  // particle functions
  void calcParticleForceEnergy();

  //void calcParticleBoundaryForceEnergy();

  void makeExternalParticleForce(double externalForce);

  void addConstantParticleForce(double externalForce, long maxIndex);

  void addExternalParticleForce();
  thrust::host_vector<double> getExternalParticleForce();

  double getParticleTotalForceMagnitude();

  double getParticleMaxUnbalancedForce();

  void calcParticleStressTensor();

  double getParticlePressure();

  double getParticleSurfaceTension();

  double getParticleShearStress();

  double getParticleExtensileStress();

  double getParticleWallForce(double range);

  double getParticleActiveWallForce(double range, double driving);

  long getTotalParticleWallCount();

  double getWallForceFromVel(double range, double timeStep);

  double getParticleWallPressure();

  double getParticleActivePressure(double driving);

  double getParticlePotentialEnergy();

  double getParticleKineticEnergy();

  double getParticleEnergy();

  double getParticleTemperature();

  double getMassiveTemperature(long firstIndex, double mass);

  double getParticleDrift();

  // contacts and neighbors
  thrust::host_vector<long> getParticleNeighbors();

  void calcParticleNeighborList(double cutDistance);

  void syncParticleNeighborsToDevice();

  void calcParticleBoxNeighborList(double cutDistance);

  void calcParticleContacts(double gapSize);

  thrust::host_vector<long> getContactVectors(double gapSize);

  // minimizers
  void initFIRE(std::vector<double> &FIREparams, long minStep, long maxStep, long numDOF);

  void setParticleMassFIRE();

  void setTimeStepFIRE(double timeStep);

  void particleFIRELoop();

  void computeParticleDrift();

  void conserveParticleMomentum();

  // NVT integrators
  void initSoftParticleLangevin(double Temp, double gamma, bool readState);

  void softParticleLangevinLoop();

  void initSoftParticleLangevinSubSet(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel);

  void softParticleLangevinSubSetLoop();

  void initSoftParticleLangevinExtField(double Temp, double gamma, bool readState);

  void softParticleLangevinExtFieldLoop();

  void initSoftParticleLangevinPerturb(double Temp, double gamma, double extForce, long firstIndex, bool readState);

  void softParticleLangevinPerturbLoop();

  void initSoftParticleLangevinFlow(double Temp, double gamma, bool readState);

  void softParticleLangevinFlowLoop();

  void initSoftParticleFlow(double gamma, bool readState);

  void softParticleFlowLoop();

  // NVE integrators
  void initSoftParticleNVE(double Temp, bool readState);

  void softParticleNVELoop();

  // Active integrators
  void initSoftParticleActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleActiveLangevinLoop();

  void initSoftParticleActiveSubSet(double Temp, double Dr, double driving, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel);

  void softParticleActiveSubSetLoop();

  void initSoftParticleActiveExtField(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleActiveExtFieldLoop();

};

#endif /* SP2D_H */
