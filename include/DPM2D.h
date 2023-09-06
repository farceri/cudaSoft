//
// Author: Francesco Arceri
// Date:   10-01-2021
//
// HEADER FILE FOR DPM2D CLASS

#ifndef DPM2D_H
#define DPM2D_H

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

// pointer-to-member function call macro
#define CALL_MEMBER_FN(object, ptrToMember) ((object).*(ptrToMember))

class DPM2D;
class FIRE;
class SimInterface;
//typedef void (DPM2D::*dpm2d)(void);

class DPM2D
{
public:

  // constructor and deconstructor
  DPM2D(long nParticles, long dim, long nVertexPerParticle);
  ~DPM2D();

  // Simulator
  FIRE * fire_;
  SimInterface * sim_;

  // variables for CUDA runtime details
  long dimGrid, dimBlock, partDimGrid;

  // dpm packing constants
  long nDim;
  long numParticles;
  long numVertexPerParticle;
  long numVertices;
  // the size distribution is defined by the number of vertices in the particles
  thrust::device_vector<long> d_numVertexInParticleList;
  thrust::device_vector<long> d_firstVertexInParticleId;
  // store the index of which particle each vertex belongs to
  thrust::device_vector<long> d_particleIdList;

  thrust::device_vector<double> d_boxSize;

  // time step
  double dt;
  // dimensional factor
  double rho0;
  double vertexRad;
  // vertex/particle energy consts
  double calA0;
  double ea; // area
  double el; // segment
  double eb; // bending
  double ec; // interaction
  // attraction constants
  double l1, l2;
  // Lennard-Jones constants
  double LJcutoff, LJecut;

  // vertex shape variables
  thrust::device_vector<double> d_l0;
  thrust::device_vector<double> d_theta0;
  thrust::device_vector<double> d_rad;
  thrust::device_vector<double> d_length;
  thrust::device_vector<double> d_l0Vel;
  // particle shape variables
  thrust::device_vector<double> d_a0;
  thrust::device_vector<double> d_area;
  thrust::device_vector<double> d_perimeter;
  thrust::device_vector<double> d_particlePos;
  thrust::device_vector<double> d_particleRad;
  thrust::device_vector<double> d_particleAngle;

  // dynamical variables
  thrust::device_vector<double> d_pos;
  thrust::device_vector<double> d_vel;
  thrust::device_vector<double> d_force;
  thrust::device_vector<double> d_energy;
  thrust::device_vector<double> d_torque;
  thrust::device_vector<double> d_lastPos;
  thrust::device_vector<double> d_disp;
  // dynamical particle variables
  thrust::device_vector<double> d_particleVel;
  thrust::device_vector<double> d_particleForce;
  thrust::device_vector<double> d_particleEnergy;
  thrust::device_vector<double> d_particleAngvel;
  thrust::device_vector<double> d_particleTorque;
  // stress
  thrust::device_vector<double> d_stress;
  thrust::device_vector<double> d_perParticleStress;

  // correlation variables
  thrust::device_vector<double> d_initialPos;
  thrust::device_vector<double> d_delta;
  thrust::device_vector<double> d_particleInitPos;
  thrust::device_vector<double> d_particleInitAngle;
	thrust::device_vector<double> d_particleDelta;
	thrust::device_vector<double> d_particleDeltaAngle;
  thrust::device_vector<double> d_particlePreviousPos;
  thrust::device_vector<double> d_particleDisp;

  //contact list
  thrust::device_vector<long> d_numContacts;
  thrust::device_vector<long> d_contactList;
  thrust::device_vector<double> d_contactVectorList;
  long maxContacts;
  long contactLimit;

  // neighbor list
  thrust::device_vector<long> d_neighborList;
  thrust::device_vector<long> d_maxNeighborList;
  long maxNeighbors;
	long neighborListSize;
	// particle neighbor list
  thrust::device_vector<long> d_numPartNeighbors;
  thrust::device_vector<long> d_partNeighborList;
  thrust::device_vector<long> d_partMaxNeighborList;
  long partMaxNeighbors;
	long partNeighborListSize;
  long neighborLimit;

  void initParticleVariables(long numParticles_);

  void initParticleDynamicalVariables(long numParticles_);

  void initRotationalVariables(long numVertices_, long numParticles_);

  void initVertexVariables(long numVertices_);

  void initDynamicalVariables(long numVertices_);

  void initDeltaVariables(long numVertices_, long numParticles_);

  void initContacts(long numParticles_);

  void initNeighbors(long numVertices_);

  void initParticleNeighbors(long numParticles_);

  void initParticleIdList();

  //setters and getters
  void setDimBlock(long dimBlock_);
  long getDimBlock();

  void setNDim(long nDim_);
  long getNDim();

  void setNumParticles(long numParticles_);
	long getNumParticles();

	void setNumVertices(long numVertices_);
	long getNumVertices();

  void setNumVertexPerParticle(long numVertexPerParticle_);
	long getNumVertexPerParticle();

  void setNumVertexInParticleList(thrust::host_vector<long> &numVertexInParticleList_);

  thrust::host_vector<long> getNumVertexInParticleList();

  void setLengthScale();

  void setParticleLengthScale();

  void setLengthScaleToOne();

  void setBoxSize(thrust::host_vector<double> &boxSize_);
  thrust::host_vector<double> getBoxSize();

  // shape variables
  void setVertexRadii(thrust::host_vector<double> &rad_);
  thrust::host_vector<double> getVertexRadii();

  double getMaxRadius();

  void setRestAreas(thrust::host_vector<double> &a0_);
  thrust::host_vector<double> getRestAreas();

  void setRestLengths(thrust::host_vector<double> &l0_);
  thrust::host_vector<double> getRestLengths();

  void setRestAngles(thrust::host_vector<double> &theta0_);
  thrust::host_vector<double> getRestAngles();

  thrust::host_vector<double> getSegmentLengths();

  void setAreas(thrust::host_vector<double> &area_);
  thrust::host_vector<double> getAreas();

  thrust::host_vector<double> getPerimeters();

  void calcParticlesShape();

  void calcParticlesPositions();

  void setDefaultParticleRadii();

  void setParticleRadii(thrust::host_vector<double> &particleRad_);
  thrust::host_vector<double> getParticleRadii();

  void setParticlePositions(thrust::host_vector<double> &particlePos_);
  void setPBCParticlePositions(thrust::host_vector<double> &particlePos_);
  thrust::host_vector<double> getParticlePositions();
  thrust::host_vector<double> getPBCParticlePositions();
  thrust::host_vector<double> getParticleDeltas();

  void resetPreviousPositions();

  void resetLastPositions();

  thrust::host_vector<double> getPreviousPositions();

  void setParticleVelocities(thrust::host_vector<double> &particleVel_);
  thrust::host_vector<double> getParticleVelocities();

  void setParticleForces(thrust::host_vector<double> &particleForce_);
  thrust::host_vector<double> getParticleForces();

  thrust::host_vector<double> getParticleEnergies();

  double getMeanParticleSize();

  double getMeanParticleSigma();

  double getMinParticleSigma();

  double getMaxDisplacement();

  void setParticleAngles(thrust::host_vector<double> &particleAngle_);
  thrust::host_vector<double> getParticleAngles();

  // dynamical variables
  void setVertexPositions(thrust::host_vector<double> &pos_);
  thrust::host_vector<double> getVertexPositions();

  void setVertexVelocities(thrust::host_vector<double> &vel_);
	thrust::host_vector<double> getVertexVelocities();

  void setVertexForces(thrust::host_vector<double> &force_);
	thrust::host_vector<double> getVertexForces();

  void setVertexTorques(thrust::host_vector<double> &torque_);
  thrust::host_vector<double> getVertexTorques();

  thrust::host_vector<double> getPerParticleStressTensor();

  thrust::host_vector<double> getStressTensor();

  double getPressure();

  double getTotalForceMagnitude();

  double getMaxUnbalancedForce();

  thrust::host_vector<long> getMaxNeighborList();

  thrust::host_vector<long> getNeighbors();

  thrust::host_vector<long> getContacts();

  void printNeighbors();

  void printContacts();

  double getPotentialEnergy();

  double getSmoothPotentialEnergy();

  double getKineticEnergy();

  double getTemperature();

  double getTotalEnergy();

  double getPhi();

  double getPreferredPhi();

  double getParticlePhi();

  double get3DParticlePhi();

  double getVertexMSD();

  double getParticleMSD();

  double getParticleMaxDisplacement();

  double getDeformableWaveNumber();

  double getSoftWaveNumber();

  double getVertexISF();

  double getParticleISF(double waveNumber_);

  double getHexaticOrderParameter();

  double getAreaFluctuation();

  // initialization functions
  void setMonoSizeDistribution();

  //void setBiSizeDistribution();

  void setPolyRandomSoftParticles(double phi0, double polyDispersity);

  void setPolySizeDistribution(double calA0, double polyDispersity);

  void setSinusoidalRestAngles(double thetaA, double thetaK);

  void setRandomParticles(double phi0, double extraRad);

  void initVerticesOnParticles();

  void scaleVertices(double scale);

  void scaleParticles(double scale);

  void pressureScaleParticles(double pscale);

  void scaleSoftParticles(double scale);

  void scaleParticleVelocity(double scale);

  void translateVertices();

  void rotateVertices();

  void computeParticleAngleFromVel();

  // constant setters
  void setEnergyCosts(double ea_, double el_, double eb_, double ec_);

  void setAttractionConstants(double l1_, double l2_);

  void setLJcutoff(double LJcutoff_);

  double setTimeScale(double dt_);

  double setTimeStep(double dt_);

  // integration functions
  void calcForceEnergy();

  void calcVertexForceAngAcc();

  void calcRigidForceEnergy();

  void calcVertexForceTorque();

  void calcRigidForceTorque();

  void calcStressTensor();

  void calcPerParticleStressTensor();

  void calcNeighborForces();

  void calcParticleNeighbors();

  void calcContacts(double gapSize);

  thrust::host_vector<long> getContactVectors(double gapSize);

  void calcNeighborList(double cutDistance);

  void syncNeighborsToDevice();

  // particle functions
  void calcParticleForceEnergy();

  void calcParticleWallForceEnergy();

  void calcParticleSidesForceEnergy();

  void calcParticleForceEnergyRA();

  void calcParticleForceEnergyLJ();

  void makeExternalParticleForce(double externalForce);

  void addExternalParticleForce();

  thrust::host_vector<double> getExternalParticleForce();

  double getParticleTotalForceMagnitude();

  double getParticleMaxUnbalancedForce();

  double getRigidMaxUnbalancedForce();

  void calcParticleStressTensor();

  double getParticleVirialPressure();

  double getParticleWallPressure();

  double getParticleDynamicalPressure();

  double getParticleActivePressure(double driving);

  double getParticleTotalPressure(double driving);

  double getParticleEnergy();

  double getParticleKineticEnergy();

  double getParticleDrift();

  double getParticleTemperature();

  double getMassiveTemperature(long firstIndex, double mass);

  thrust::host_vector<long> getParticleNeighbors();

  void calcParticleNeighborList(double cutDistance);

  void calcParticleWallNeighborList(double cutDistance);

  void syncParticleNeighborsToDevice();

  void calcParticleContacts(double gapSize);

  // minimizers
  void initFIRE(std::vector<double> &FIREparams, long minStep, long maxStep, long numDOF);

  void setParticleMassFIRE();

  void setTimeStepFIRE(double timeStep);

  void particleFIRELoop();

  void vertexFIRELoop();

  void initRigidFIRE(std::vector<double> &FIREparams, long minStep, long numStep, long numDOF, double cutDist);

  void rigidFIRELoop();

  // simulators for deformable particles
  void initLangevin(double Temp, double gamma, bool readState);

  void langevinLoop();

  void initActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState);

  void activeLangevinLoop();

  void initNVE(double Tin, bool readState);

  void NVELoop();

  void initBrownian(double Temp, double gamma, bool readState);

  void brownianLoop();

  void initActiveBrownian(double Dr, double driving, bool readState);

  void activeBrownianLoop();

  void initActiveBrownianDampedL0(double Dr, double driving, double gamma, bool readState);

  void activeBrownianDampedL0Loop();

  // simulators for soft particles
  void computeParticleDeltas();

  void computeParticleDrift();

  void conserveParticleMomentum();

  void initSoftParticleLangevin(double Temp, double gamma, bool readState);

  void softParticleLangevinLoop();

  void initSoftParticleLangevinFixedBoundary(double Temp, double gamma, bool readState);

  void softParticleLangevinFixedBoundaryLoop();

  void initSoftParticleNVE(double Temp, bool readState);

  void softParticleNVELoop();

  void initSoftParticleNVERA(double Temp, bool readState);

  void softParticleNVERALoop();

  void initSoftParticleNVEFixedBoundary(double Temp, bool readState);

  void softParticleNVEFixedBoundaryLoop();

  void initSoftParticleActiveNVEFixedBoundary(double Temp, double Dr, double driving, bool readState);

  void softParticleActiveNVEFixedBoundaryLoop();

  void initSoftParticleLangevinRA(double Temp, double gamma, bool readState);

  void softParticleLangevinRALoop();

  void initSoftParticleLangevinLJ(double Temp, double gamma, bool readState);

  void softParticleLangevinLJLoop();

  void initSoftLangevinSubSet(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel);

  void softLangevinSubSetLoop();

  void initSoftParticleLExtField(double Temp, double gamma, bool readState);

  void softParticleLExtFieldLoop();

  void initSoftParticleActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleActiveLangevinLoop();

  void initSoftParticleActiveLangevinFixedBoundary(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleActiveLangevinFixedBoundaryLoop();

  void initSoftParticleActiveLangevinFixedSides(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleActiveLangevinFixedSidesLoop();

  void initSoftALSubSet(double Temp, double Dr, double driving, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel);

  void softALSubSetLoop();

  void initSoftParticleALExtField(double Temp, double Dr, double driving, double gamma, bool readState);

  void softParticleALExtFieldLoop();

  // simulators for rigid particles
  void initRigidBrownian(double Temp, double cutDistance, bool readState);

  void rigidBrownianLoop();

  void initRigidRotActiveBrownian(double Dr, double driving, double cutDistance, bool readState);

  void rigidRotActiveBrownianLoop();

  void initRigidActiveBrownian(double Dr, double driving, double cutDistance, bool readState);

  void rigidActiveBrownianLoop();

};

#endif /* DPM2D_H */
