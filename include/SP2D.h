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
  enum class geometryEnum {normal, leesEdwards, fixedBox, fixedSides2D, fixedSides3D, roundBox} geometryType;
  enum class neighborEnum {neighbor, allToAll} neighborType;
  enum class potentialEnum {harmonic, lennardJones, Mie, WCA, adhesive, doubleLJ, LJMinusPlus, LJWCA} potentialType;
  enum class boxEnum {harmonic, WCA, reflect, reflectnoise} boxType;
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
  thrust::device_vector<double> d_particlePos;
  thrust::device_vector<double> d_particleRad;
  thrust::device_vector<double> d_particleVel;
  thrust::device_vector<double> d_particleLastVel;
  thrust::device_vector<double> d_particleForce;
  thrust::device_vector<double> d_particleEnergy;
  thrust::device_vector<double> d_particleAngle;
  thrust::device_vector<double> d_particleOmega;
  thrust::device_vector<double> d_particleAlpha;
  thrust::device_vector<double> d_activeAngle;
  thrust::device_vector<double> d_randomAngle;
  thrust::device_vector<double> d_stress;
  thrust::device_vector<double> d_wallForce;
  thrust::device_vector<long> d_wallCount;
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
  thrust::device_vector<long> d_partNeighborList;
  thrust::device_vector<long> d_partMaxNeighborList;
  long partMaxNeighbors;
	long partNeighborListSize;
  long neighborLimit;

  // Vicsek interaction list
  thrust::device_vector<long> d_vicsekNeighborList;
  thrust::device_vector<long> d_vicsekMaxNeighborList;
  long vicsekMaxNeighbors;
	long vicsekNeighborListSize;
  long vicsekNeighborLimit;

  double checkGPUMemory();

  void initParticleVariables(long numParticles_);

  void initParticleDeltaVariables(long numParticles_);

  void initContacts(long numParticles_);

  void initParticleNeighbors(long numParticles_);

  void initVicsekNeighbors(long numParticles_);

  //setters and getters
  void syncSimControlToDevice();
  void syncSimControlFromDevice();
  bool testSimControlSync();

  void setParticleType(simControlStruct::particleEnum particleType_);
	simControlStruct::particleEnum getParticleType();

  void setGeometryType(simControlStruct::geometryEnum geometryType_);
	simControlStruct::geometryEnum getGeometryType();

  void setNeighborType(simControlStruct::neighborEnum neighborType_);
	simControlStruct::neighborEnum getNeighborType();

  void setPotentialType(simControlStruct::potentialEnum potentialType_);
	simControlStruct::potentialEnum getPotentialType();

  void setBoxType(simControlStruct::boxEnum boxType_);
	simControlStruct::boxEnum getBoxType();

  void setGravityType(simControlStruct::gravityEnum gravityType_);
	simControlStruct::gravityEnum getGravityType();

  void setLEshift(double LEshift_);
  double getLEshift();

  void applyLEShear(double LEshift_);

  void applyExtension(double strainy_);

  void applyUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_);

  void applyCenteredUniaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_);

  void applyBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_);

  void applyBiaxialExpExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_);

  void applyCenteredBiaxialExtension(thrust::host_vector<double> &newBoxSize_, double strain_, long direction_);

  void setDimBlock(long dimBlock_);
  long getDimBlock();

  void setNDim(long nDim_);
  long getNDim();

  void setNumParticles(long numParticles_);
	long getNumParticles();

  long getTypeNumParticles();

  void setParticleLengthScale();

  void setLengthScaleToOne();

  void setBoxSize(thrust::host_vector<double> &boxSize_);
  thrust::host_vector<double> getBoxSize();

  void setBoxRadius(double boxRadius_);
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

  thrust::host_vector<double> getParticleDeltas();

  void resetLastPositions();

  void resetVicsekLastPositions();

  void setInitialPositions();

  thrust::host_vector<double> getLastPositions();

  void resetLastVelocities();

  void setParticleVelocities(thrust::host_vector<double> &particleVel_);
  thrust::host_vector<double> getParticleVelocities();

  void setParticleForces(thrust::host_vector<double> &particleForce_);
  thrust::host_vector<double> getParticleForces();

  thrust::host_vector<double> getParticleEnergies();

  void setParticleAngles(thrust::host_vector<double> &particleAngle_);
  void checkParticleAngles();
  thrust::host_vector<double> getParticleAngles();

  thrust::host_vector<long> getContacts();

  void printContacts();

  double getParticlePhi();

  double getParticleMSD();

  double setDisplacementCutoff(double cutoff_);

  void resetUpdateCount();

  long getUpdateCount();

  void checkParticleNeighbors();

  void checkVicsekNeighbors();

  double getParticleMaxDisplacement();

  void checkParticleDisplacement();

  void checkParticleMaxDisplacement();

  void checkParticleMaxDisplacement2();

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

  // force and energy
  void setEnergyCostant(double ec_);
  double getEnergyCostant();

  double setTimeStep(double dt_);

  void setSelfPropulsionParams(double driving_, double taup_);
  void getSelfPropulsionParams(double &driving_, double &taup_);

  void setVicsekParams(double driving_, double Jvicsek_, double Rvicsek_);
  void getVicsekParams(double &driving_, double &Jvicsek_, double &Rvicsek_);

  void setAdhesionParams(double l1_, double l2_);

  void setLJcutoff(double LJcutoff_);

  void setDoubleLJconstants(double LJcutoff_, double eAA_, double eAB_, double eBB_, long num1_);

  void setLJMinusPlusParams(double LJcutoff_, long num1_);

  void setLJWCAparams(double LJcutoff_, long num1_);

  void setMieParams(double LJcutoff_, double nPower_, double mPower_);

  void setBoxEnergyScale(double ew_);

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

  void addParticleWallInteraction();

  void calcParticleWallInteraction();

  void reflectParticleOnWall();

  void reflectParticleOnWallWithNoise();

  void addParticleGravity();

  void calcParticleForceEnergy();

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

  void calcParticleActiveStressTensor();

  double getParticleActivePressure();

  double getParticleSurfaceTension();

  double getParticleShearStress();

  double getParticleExtensileStress();

  std::tuple<double, double, double> getParticleStressComponents();

  double getParticleWallPressure();

  double getParticleBoxPressure();

  std::tuple<double, double> getColumnWork(double width);
  
  std::tuple<double, double> getColumnActiveWork(double width);

  double getParticleWallForce(double range, double width);

  long getTotalParticleWallCount();

  double getParticlePotentialEnergy();

  double getParticleKineticEnergy();

  double getDampingWork();

  double getSelfPropulsionWork();

  double getParticleEnergy();

  double getParticleWork();

  double getParticleTemperature();

  void adjustKineticEnergy(double prevEtot);

  void adjustLocalKineticEnergy(thrust::host_vector<double> &prevEnergy_, long direction_);

  void adjustTemperature(double targetTemp);

  std::tuple<double, double, double> getParticleT1T2();

  std::tuple<double, double, double> getParticleKineticEnergy12();

  double getMassiveTemperature(long firstIndex, double mass);

  void removeCOMDrift();

  // contacts and neighbors
  thrust::host_vector<long> getParticleNeighbors();

  thrust::host_vector<long> getVicsekNeighbors();

  void calcParticleNeighbors(double cutDistance);

  void calcParticleNeighborList(double cutDistance);

  void syncParticleNeighborsToDevice();

  void calcVicsekNeighborList();

  void syncVicsekNeighborsToDevice();

  void calcParticleContacts(double gapSize);

  thrust::host_vector<long> getContactVectors(double gapSize);

  // minimizers
  void initFIRE(std::vector<double> &FIREparams, long minStep, long maxStep, long numDOF);

  void setParticleMassFIRE();

  void setTimeStepFIRE(double timeStep);

  void particleFIRELoop();

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
