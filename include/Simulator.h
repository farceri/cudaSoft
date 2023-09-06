//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//
// HEADER FILE FOR INTEGRATOR CLASS
// We define different integrator as child classes of SimulatorInterface where
// all the essential integration functions are defined

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "DPM2D.h"

class DPM2D;

class SimConfig // initializer
{
public:
  double Tinject;
  double Dr;
  double driving;
  double cutDist;
  SimConfig() = default;
  SimConfig(double Tin, double Dr, double driving, double cutDist):Tinject(Tin), Dr(Dr), driving(driving), cutDist(cutDist){}
};

class SimInterface // integration functions
{
public:
  DPM2D * dpm_;
  SimConfig config;
  double lcoeff1;
  double lcoeff2;
  double lcoeff3;
  double noiseVar;
  double gamma = 1; // this is just a choice
  long firstIndex = 10;
  double mass = 1;
  thrust::device_vector<double> d_rand;
  thrust::device_vector<double> d_rando;
  thrust::device_vector<double> d_pActiveAngle; // for decoupled rotation and activity angles
  thrust::device_vector<double> d_thermalVel; // for brownian noise of soft particles

  SimInterface() = default;
  SimInterface(DPM2D * dpmPtr, SimConfig config):dpm_(dpmPtr),config(config){}
  ~SimInterface();

  virtual void injectKineticEnergy() = 0;
  virtual void updatePosition(double timeStep) = 0;
  virtual void updateVelocity(double timeStep) = 0;
  virtual void updateThermalVel() = 0;
  virtual void conserveMomentum() = 0;
  virtual void integrate() = 0;
};

//****************** integrators for deformable particles ********************//
// Langevin integrator child of SimulatorInterface
class Langevin: public SimInterface
{
public:
  Langevin() = default;
  Langevin(DPM2D * dpmPtr, SimConfig config) : SimInterface:: SimInterface(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Langevin2 integrator child of Langevin
class Langevin2: public Langevin
{
public:
  Langevin2() = default;
  Langevin2(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Active Langevin integrator child of Langevin2
class ActiveLangevin: public Langevin2
{
public:
  ActiveLangevin() = default;
  ActiveLangevin(DPM2D * dpmPtr, SimConfig config) : Langevin2:: Langevin2(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// NVE integrator child of Langevin
class NVE: public Langevin
{
public:
  NVE() = default;
  NVE(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void integrate();
};

// Brownian integrator child of NVE
class Brownian: public NVE
{
public:
  Brownian() = default;
  Brownian(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Active Brownian integrator child of NVE
class ActiveBrownian: public NVE
{
public:
  ActiveBrownian() = default;
  ActiveBrownian(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Active Brownian integrator with damping on l0 child of NVE
class ActiveBrownianDampedL0: public NVE
{
public:
  ActiveBrownianDampedL0() = default;
  ActiveBrownianDampedL0(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

//********************* integrators for soft particles ***********************//
// Soft particle Langevin integrator child of SimInterface
class SoftParticleLangevin: public SimInterface
{
public:
  SoftParticleLangevin() = default;
  SoftParticleLangevin(DPM2D * dpmPtr, SimConfig config) : SimInterface:: SimInterface(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVE: public SoftParticleLangevin
{
public:
  SoftParticleNVE() = default;
  SoftParticleNVE(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void integrate();
};

// Attractive soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVERA: public SoftParticleLangevin
{
public:
  SoftParticleNVERA() = default;
  SoftParticleNVERA(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void integrate();
};

// Fixed boundary soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVEFixedBoundary: public SoftParticleLangevin
{
public:
  SoftParticleNVEFixedBoundary() = default;
  SoftParticleNVEFixedBoundary(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleActiveNVEFixedBoundary: public SoftParticleLangevin
{
public:
  SoftParticleActiveNVEFixedBoundary() = default;
  SoftParticleActiveNVEFixedBoundary(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void updateThermalVel();

  virtual void integrate();
};

// Soft particle Langevin integrator child of SimInterface
class SoftParticleLangevin2: public SoftParticleLangevin
{
public:
  SoftParticleLangevin2() = default;
  SoftParticleLangevin2(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Langevin integrator with repulsive and attractive forces child of SoftParticleLangevin2
class SoftParticleLangevin2RA: public SoftParticleLangevin2
{
public:
  SoftParticleLangevin2RA() = default;
  SoftParticleLangevin2RA(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with repulsive and attractive forces child of SoftParticleLangevin2
class SoftParticleLangevin2LJ: public SoftParticleLangevin2
{
public:
  SoftParticleLangevin2LJ() = default;
  SoftParticleLangevin2LJ(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with repulsive and attractive forces child of SoftParticleLangevin2
class SoftParticleLangevinFixedBoundary: public SoftParticleLangevin2
{
public:
  SoftParticleLangevinFixedBoundary() = default;
  SoftParticleLangevinFixedBoundary(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with massive particles child of SoftParticleLangevin2
class SoftLangevinSubSet: public SoftParticleLangevin
{
public:
  SoftLangevinSubSet() = default;
  SoftLangevinSubSet(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Soft particle Langevin integrator with external field child of softParticleLangevin2
class SoftParticleLExtField: public SoftParticleLangevin2
{
public:
  SoftParticleLExtField() = default;
  SoftParticleLExtField(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleLangevin2
class SoftParticleActiveLangevin: public SoftParticleLangevin2
{
public:
  SoftParticleActiveLangevin() = default;
  SoftParticleActiveLangevin(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleActiveLangevin
class SoftParticleActiveLangevinFixedBoundary: public SoftParticleActiveLangevin
{
public:
  SoftParticleActiveLangevinFixedBoundary() = default;
  SoftParticleActiveLangevinFixedBoundary(DPM2D * dpmPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleActiveLangevin
class SoftParticleActiveLangevinFixedSides: public SoftParticleActiveLangevin
{
public:
  SoftParticleActiveLangevinFixedSides() = default;
  SoftParticleActiveLangevinFixedSides(DPM2D * dpmPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(dpmPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator with massive particles child of SoftParticleActiveLangevin
class SoftALSubSet: public SoftLangevinSubSet
{
public:
  SoftALSubSet() = default;
  SoftALSubSet(DPM2D * dpmPtr, SimConfig config) : SoftLangevinSubSet:: SoftLangevinSubSet(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Active Langevin integrator with external field child of SoftParticleActiveLangevin
class SoftParticleALExtField: public SoftParticleActiveLangevin
{
public:
  SoftParticleALExtField() = default;
  SoftParticleALExtField(DPM2D * dpmPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(dpmPtr, config){;}

  virtual void integrate();
};


//********************* integrators for rigid particles **********************//
// Rigid Brownian integrator child of SofParicleLangevin
class RigidBrownian: public SoftParticleLangevin
{
public:
  RigidBrownian() = default;
  RigidBrownian(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Rigid Active Brownian integrator child of RigidBrownian
class RigidActiveBrownian: public RigidBrownian
{
public:
  RigidActiveBrownian() = default;
  RigidActiveBrownian(DPM2D * dpmPtr, SimConfig config) : RigidBrownian:: RigidBrownian(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Rigid Rotational Active Brownian integrator child of SoftParticleLangevin
class RigidRotActiveBrownian: public SoftParticleLangevin
{
public:
  RigidRotActiveBrownian() = default;
  RigidRotActiveBrownian(DPM2D * dpmPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

#endif // SIMULATOR_H //
