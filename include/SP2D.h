//
// Author: Francesco Arceri
// Date:   10-01-2021
//
// HEADER FILE FOR SP2D CLASS

#ifndef SP2D_H
#define SP2D_H

#include "defs.h"
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using std::vector;
using std::string;
using std::tuple;

struct simControlStruct {
  enum class particleEnum {passive, active, vicsek} particleType;
  enum class noiseEnum {langevin1, langevin2, brownian, drivenBrownian} noiseType;
  enum class boundaryEnum {pbc, leesEdwards, fixed, reflect, reflectNoise, rough, rigid, mobile, plastic} boundaryType;
  enum class geometryEnum {squareWall, fixedSides2D, fixedSides3D, roundWall} geometryType;
  enum class neighborEnum {neighbor, allToAll} neighborType;
  enum class potentialEnum {none, harmonic, lennardJones, Mie, WCA, adhesive, doubleLJ, LJMinusPlus, LJWCA} potentialType;
  enum class wallEnum {harmonic, lennardJones, WCA} wallType;
  enum class gravityEnum {on, off} gravityType;
  enum class alignEnum {additive, nonAdditive, velAlign} alignType;
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
  double boxRadius;

  // time step
  double dt;
  // dimensional factor
  double rho0;
  // energy scale
  double ec;
  // self-propulsion parameters
  double driving, taup;
  // Vicsek velocity interaction parameters
  double Jvicsek, Rvicsek;
  // Reflection noise
  double angleAmplitude;
  // adhesion constants
  double l1, l2;
  // Lennard-Jones constants
  double LJcutoff, LJecut, LJfshift;
  double LJecutPlus, LJfshiftPlus;
  // double Lennard-Jones constants
  double eAA, eAB, eBB;
  long num1;
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
  double cutoff, cutDistance, rcut;
  long updateCount;
  bool shift;

  // dynamical particle variables
  thrust::device_vector<double> d_particleRad;
  thrust::device_vector<double> d_particlePos;
  thrust::device_vector<double> d_particleVel;
  thrust::device_vector<double> d_squaredVel;
  thrust::device_vector<double> d_particleForce;
  thrust::device_vector<double> d_particleEnergy;
  thrust::device_vector<double> d_particleAngle;
  thrust::device_vector<double> d_particleOmega;
  thrust::device_vector<double> d_particleAlpha;
  thrust::device_vector<double> d_randAngle;
  thrust::device_vector<double> d_randomAngle;
  thrust::device_vector<double> d_stress;
  // wall variables
  thrust::device_vector<double> d_wallLength;
  thrust::device_vector<double> d_restLength;
  thrust::device_vector<double> d_wallAngle;
  thrust::device_vector<double> d_areaSector;
  thrust::device_vector<double> d_wallPos;
  thrust::device_vector<double> d_wallVel;
  thrust::device_vector<double> d_wallForce;
  thrust::device_vector<double> d_wallEnergy;
  thrust::device_vector<double> d_sqWallVel;
  thrust::device_vector<long> d_wallCount;
  long numWall;
  // mobile wall
  double wallRad;
  double wallArea;
  double wallArea0;
  double wallLength0;
  double wallAngle0;
  double ea, el, eb;
  // plastic wall
  double lgamma;
  // rigid wall
  double wallAngle;
  double wallOmega;
  double wallAlpha;
  thrust::device_vector<double> d_monomerAlpha;
  // correlation variables
  thrust::device_vector<double> d_velCorr;
  thrust::device_vector<double> d_unitPos;
  thrust::device_vector<double> d_unitVel;
  thrust::device_vector<double> d_unitVelPos;
  thrust::device_vector<double> d_angMom;
  // two-type particles variables
  thrust::device_vector<long> d_flagAB;
  thrust::device_vector<double> d_squaredVelAB;
  thrust::device_vector<double> d_particleEnergyAB;
  // hydrodynamical variables
  thrust::device_vector<double> d_flowVel;
  thrust::device_vector<double> d_surfaceHeight;

  // correlation variables
  thrust::device_vector<double> d_particleInitPos;
  thrust::device_vector<double> d_particleLastPos;
  thrust::device_vector<double> d_vicsekLastPos;
	thrust::device_vector<double> d_particleDelta;
  thrust::device_vector<double> d_particleDisp;

  // contact list
  thrust::device_vector<long> d_numContacts;
  thrust::device_vector<long> d_contactList;
  thrust::device_vector<double> d_contactVectorList;
  long maxContacts;
  long contactLimit;

	// neighbor list
  thrust::device_vector<int> d_flag;
  thrust::device_vector<long> d_partNeighborList;
  thrust::device_vector<long> d_partMaxNeighborList;
  long partMaxNeighbors;
	long partNeighborListSize;
  long neighborLimit;

  // Vicsek interaction list
  thrust::device_vector<int> d_vicsekFlag;
  thrust::device_vector<long> d_vicsekNeighborList;
  thrust::device_vector<long> d_vicsekMaxNeighborList;
  long vicsekMaxNeighbors;
	long vicsekNeighborListSize;
  long vicsekNeighborLimit;

  // Wall interaction list
  thrust::device_vector<long> d_wallNeighborList;
  thrust::device_vector<long> d_wallMaxNeighborList;
  long wallMaxNeighbors;
	long wallNeighborListSize;
  long wallNeighborLimit;

  double checkGPUMemory();

  void initParticleVariables(long numParticles_);

  void initParticleDeltaVariables(long numParticles_);

  void initContacts(long numParticles_);

  void initParticleNeighbors(long numParticles_);

  void initVicsekNeighbors(long numParticles_);

  void initWallVariables(long numWall_);

  void initWallShapeVariables(long numWall_);

  void initWallNeighbors(long numParticles_);

  //setters and getters
  void syncSimControlToDevice();
  void syncSimControlFromDevice();
  bool testSimControlSync();

  void setParticleType(simControlStruct::particleEnum particleType_);

  void setNoiseType(simControlStruct::noiseEnum noiseType_);

  void setBoundaryType(simControlStruct::boundaryEnum boundaryType_);
	simControlStruct::boundaryEnum getBoundaryType();

  void setGeometryType(simControlStruct::geometryEnum geometryType_);
	simControlStruct::geometryEnum getGeometryType();

  void setNeighborType(simControlStruct::neighborEnum neighborType_);
	simControlStruct::neighborEnum getNeighborType();

  void setPotentialType(simControlStruct::potentialEnum potentialType_);
	simControlStruct::potentialEnum getPotentialType();

  void setWallType(simControlStruct::wallEnum wallType_);
	simControlStruct::wallEnum getWallType();

  void setGravityType(simControlStruct::gravityEnum gravityType_);
	simControlStruct::gravityEnum getGravityType();

  void setAlignType(simControlStruct::alignEnum alignType_);
	simControlStruct::alignEnum getAlignType();

  void setLEshift(double LEshift_);
  double getLEshift();

  void applyLEShear(double LEshift_);

  void applyExtension(double strainy_);

  void applyUniaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_);

  void applyCenteredUniaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_);

  void applyBiaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_);

  void applyBiaxialExpExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_);

  void applyCenteredBiaxialExtension(thrust::host_vector<double> &newWallSize_, double strain_, long direction_);

  void setDimBlock(long dimBlock_);
  void setWallBlock(long dimBlock_);
  long getDimBlock();

  void setNDim(long nDim_);
  long getNDim();

  void setNumParticles(long numParticles_);
	long getNumParticles();

  long getTypeNumParticles();

  void setNumWall(long numWall_);
	long getNumWall();

  double getWallRad();

  double getWallArea0();

  double getWallArea();

  double getWallAreaDeviation();

  std::tuple<double, double, double> getWallAngleDynamics();
  void setWallAngleDynamics(thrust::host_vector<double> wallDynamics_);

  void setParticleLengthScale();

  void setLengthScaleToOne();

  void setBoxSize(thrust::host_vector<double> &boxSize_);
  thrust::host_vector<double> getBoxSize();

  void setBoxRadius(double boxRadius_);
  void scaleBoxRadius(double scale_);
  double getBoxRadius();

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

  void resetLastPositions();

  void resetVicsekLastPositions();

  void setInitialPositions();

  thrust::host_vector<double> getLastPositions();

  void setParticleVelocities(thrust::host_vector<double> &particleVel_);
  thrust::host_vector<double> getParticleVelocities();

  void setParticleForces(thrust::host_vector<double> &particleForce_);
  thrust::host_vector<double> getParticleForces();

  thrust::host_vector<double> getWallForces();

  thrust::host_vector<double> getParticleEnergies();

  void setParticleAngles(thrust::host_vector<double> &particleAngle_);
  thrust::host_vector<double> getParticleAngles();

  void setWallPositions(thrust::host_vector<double> &wallPos_);
  thrust::host_vector<double> getWallPositions();

  void setWallVelocities(thrust::host_vector<double> &wallVel_);
  thrust::host_vector<double> getWallVelocities();

  void setWallLengths(thrust::host_vector<double> &wallLength_);
  thrust::host_vector<double> getWallLengths();

  void setWallAngles(thrust::host_vector<double> &wallAngle_);
  thrust::host_vector<double> getWallAngles();

  thrust::host_vector<long> getContacts();

  void printContacts();

  double getParticlePhi();

  double getParticleMSD();

  double setDisplacementCutoff(double cutoff_);

  void resetUpdateCount();

  long getUpdateCount();

  double getParticleMaxDisplacement();

  void checkParticleDisplacement();

  void checkParticleMaxDisplacement();

  void checkParticleMaxDisplacement2();

  void checkParticleNeighbors();

  void checkVicsekNeighbors();

  double getSoftWaveNumber();

  double getParticleISF(double waveNumber_);

  // initialization functions
  void setPolyRandomParticles(double phi0, double polyDispersity);

  void setScaledPolyRandomParticles(double phi0, double polyDispersity, double lx, double ly, double lz);

  void setRoundScaledPolyRandomParticles(double phi0, double polyDispersity, double boxRadius_);

  void setScaledMonoRandomParticles(double phi0, double lx, double ly, double lz);

  void setScaledBiRandomParticles(double phi0, double lx, double ly, double lz);

  void pressureScaleParticles(double pscale);

  void scaleParticles(double scale);

  void scaleParticlePacking();

  void scaleParticleVelocity(double scale);

  void initializeParticleAngles();

  void setWallShapeEnergyScales(double ea_, double el_, double eb_);

  void setMobileWallParams(long numWall_, double wallRad_, double wallArea0_);

  void setPlasticVariables(double lgamma_);

  void initRigidWall();

  void checkDimGrid();

  void initMobileWall();

  void initializeWall();

  // force and energy
  void setEnergyCostant(double ec_);
  double getEnergyCostant();

  double setTimeStep(double dt_);

  void setSelfPropulsionParams(double driving_, double taup_);
  void getSelfPropulsionParams(double &driving_, double &taup_);

  void setVicsekParams(double driving_, double taup_, double Jvicsek_, double Rvicsek_);
  void getVicsekParams(double &driving_, double &taup_, double &Jvicsek_, double &Rvicsek_);

  void setReflectionNoise(double angleAmplitude_);

  void setAdhesionParams(double l1_, double l2_);

  void setLJcutoff(double LJcutoff_);

  void setDoubleLJconstants(double LJcutoff_, double eAA_, double eAB_, double eBB_, long num1_);

  void setLJMinusPlusParams(double LJcutoff_, long num1_);

  void setLJWCAparams(double LJcutoff_, long num1_);

  void setMieParams(double LJcutoff_, double nPower_, double mPower_);

  void setWallEnergyScale(double ew_);

  void setGravity(double gravity_, double ew_);

  void setFluidFlow(double speed_, double viscosity_);

  void calcSurfaceHeight();

  double getSurfaceHeight();

  void calcFlowVelocity();

  // particle functions
  void calcParticleInteraction();

  void addSelfPropulsion();

  void addVicsekAlignment();

  void calcVicsekAlignment();

  void calcParticleFixedWallInteraction();

  void calcWallArea();

  void calcWallShape();

  void calcWallShapeForceEnergy();

  void calcPlasticWallShapeForceEnergy();
  
  void calcParticleWallInteraction();

  void calcWallAngularAcceleration();

  void checkParticleInsideRoundWall();

  void checkReflectiveWall();

  void reflectParticleOnWall();

  void reflectParticleOnWallWithNoise();

  void addParticleGravity();

  void calcParticleForceEnergy();

  std::tuple<double, double, double> getVicsekOrderParameters();

  double getVicsekHigherOrderParameter(double order_);

  double getVicsekVelocityCorrelation();

  double getNeighborVelocityCorrelation();

  double getParticleAngularMomentum();

  void setTwoParticleTestPacking(double sigma0, double sigma1, double lx, double ly, double y0, double y1, double vel1);

  void setThreeParticleTestPacking(double sigma01, double sigma2, double lx, double ly, double y01, double y2, double vel2);

  void firstUpdate(double timeStep);

  void secondUpdate(double timeStep);

  void testInteraction(double timeStep);

  void printTwoParticles();

  void printThreeParticles();

  //void calcParticleBoundaryForceEnergy();

  void makeExternalParticleForce(double externalForce);

  void addConstantParticleForce(double externalForce, long maxIndex);

  void addExternalParticleForce();
  thrust::host_vector<double> getExternalParticleForce();

  double getParticleTotalForceMagnitude();

  double getParticleMaxUnbalancedForce();

  void calcParticleStressTensor();

  double getParticlePressure();

  double getParticleTotalPressure();

  void calcParticleActiveStressTensor();

  double getParticleActivePressure();

  double getParticleSurfaceTension();

  double getParticleShearStress();

  double getParticleExtensileStress();

  std::tuple<double, double, double> getParticleStressComponents();

  std::tuple<double, double> computeWallPressure();

  std::tuple<double, double> getWallPressure();

  std::tuple<double, double> getColumnWork(double width);
  
  std::tuple<double, double> getColumnActiveWork(double width);

  double getParticleWallForce(double range, double width);

  long getTotalParticleWallCount();

  double getParticlePotentialEnergy();

  double getWallPotentialEnergy();

  double getParticleKineticEnergy();

  double getWallKineticEnergy();

  double getWallRotationalKineticEnergy();

  double getDampingWork();

  double getNoiseWork();

  double getSelfPropulsionWork();

  double getParticleWork();

  double getParticleTemperature();

  double getWallTemperature();

  double getParticleEnergy();

  double getWallEnergy();

  double getTotalEnergy();

  void adjustKineticEnergy(double prevEtot);

  void adjustLocalKineticEnergy(thrust::host_vector<double> &prevEnergy_, long direction_);

  void adjustTemperature(double targetTemp);

  std::tuple<double, double, double> getParticleT1T2();

  std::tuple<double, double, double> getParticleKineticEnergy12();

  double getMassiveTemperature(long firstIndex, double mass);

  void calcParticleEnergyAB();
  
  std::tuple<double, double, long> getParticleEnergyAB();

  void calcParticleHeatAB();

  std::tuple<double, double, double, long> getParticleWorkAB();

  void removeCOMDrift();

  // contacts and neighbors
  thrust::host_vector<long> getParticleNeighbors();

  thrust::host_vector<long> getVicsekNeighbors();

  thrust::host_vector<long> getWallNeighbors();

  void calcParticleNeighbors(double cutDistance);

  void calcParticleNeighborList(double cutDistance);

  void syncParticleNeighborsToDevice();

  void calcVicsekNeighborList();

  void syncVicsekNeighborsToDevice();

  void calcWallNeighborList(double cutDistance);

  void syncWallNeighborsToDevice();

  void calcParticleContacts(double gapSize);

  thrust::host_vector<long> getContactVectors(double gapSize);

  // minimizers
  void initFIRE(std::vector<double> &FIREparams, long minStep, long maxStep, long numDOF);

  void setParticleMassFIRE();

  void setTimeStepFIRE(double timeStep);

  void particleFIRELoop();

  // Langevin integrators
  void initSoftParticleLangevin(double Temp, double gamma, bool readState);

  void softParticleLangevinLoop(bool conserve = false);

  void initSoftParticleLangevinSubset(double Temp, double gamma, long firstIndex, double mass, bool readState, bool zeroOutMassiveVel);

  void softParticleLangevinSubsetLoop();

  void initSoftParticleLangevinExtField(double Temp, double gamma, bool readState);

  void softParticleLangevinExtFieldLoop();

  void initSoftParticleLangevinPerturb(double Temp, double gamma, double extForce, long firstIndex, bool readState);

  void softParticleLangevinPerturbLoop();

  // Fluid flow integrators
  void initSoftParticleLangevinFlow(double Temp, double gamma, bool readState);

  void softParticleLangevinFlowLoop();

  void initSoftParticleFlow(double gamma, bool readState);

  void softParticleFlowLoop();

  // NVE integrators
  void initSoftParticleNVE(double Temp, bool readState);

  void softParticleNVELoop();

  void rescaleParticleVelocity(double Temp);

  void initSoftParticleNVERescale(double Temp);

  void softParticleNVERescaleLoop();

  void initSoftParticleNVEDoubleRescale(double Temp1, double Temp2);

  void softParticleNVEDoubleRescaleLoop();

  // Nose-Hoover integrator
  void getNoseHooverParams(double &mass, double &damping);

  void initSoftParticleNoseHoover(double Temp, double mass, double gamma, bool readState);

  void softParticleNoseHooverLoop();

  void getDoubleNoseHooverParams(double &mass, double &damping1, double &damping2);

  void initSoftParticleDoubleNoseHoover(double Temp1, double Temp2, double mass, double gamma1, double gamma2, bool readState);

  void softParticleDoubleNoseHooverLoop();

};

#endif /* SP2D_H */
