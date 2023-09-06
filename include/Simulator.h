//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//
// HEADER FILE FOR INTEGRATOR CLASS
// We define different integrator as child classes of SimulatorInterface where
// all the essential integration functions are defined

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "SP2D.h"

class SP2D;

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
  SP2D * sp_;
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
  SimInterface(SP2D * spPtr, SimConfig config):sp_(spPtr),config(config){}
  ~SimInterface();

  virtual void injectKineticEnergy() = 0;
  virtual void updatePosition(double timeStep) = 0;
  virtual void updateVelocity(double timeStep) = 0;
  virtual void updateThermalVel() = 0;
  virtual void conserveMomentum() = 0;
  virtual void integrate() = 0;
};

//********************* integrators for soft particles ***********************//
// Soft particle Langevin integrator child of SimInterface
class SoftParticleLangevin: public SimInterface
{
public:
  SoftParticleLangevin() = default;
  SoftParticleLangevin(SP2D * spPtr, SimConfig config) : SimInterface:: SimInterface(spPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Soft particle Langevin integrator child of SimInterface
class SoftParticleLangevin2: public SoftParticleLangevin
{
public:
  SoftParticleLangevin2() = default;
  SoftParticleLangevin2(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

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
  SoftParticleLangevin2RA(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with repulsive and attractive forces child of SoftParticleLangevin2
class SoftParticleLangevin2LJ: public SoftParticleLangevin2
{
public:
  SoftParticleLangevin2LJ() = default;
  SoftParticleLangevin2LJ(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with repulsive and attractive forces child of SoftParticleLangevin2
class SoftParticleLangevinFixedBoundary: public SoftParticleLangevin2
{
public:
  SoftParticleLangevinFixedBoundary() = default;
  SoftParticleLangevinFixedBoundary(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with massive particles child of SoftParticleLangevin2
class SoftParticleLangevinSubSet: public SoftParticleLangevin
{
public:
  SoftParticleLangevinSubSet() = default;
  SoftParticleLangevinSubSet(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Soft particle Langevin integrator with external field child of softParticleLangevin2
class SoftParticleExtField: public SoftParticleLangevin2
{
public:
  SoftParticleExtField() = default;
  SoftParticleExtField(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVE: public SoftParticleLangevin
{
public:
  SoftParticleNVE() = default;
  SoftParticleNVE(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void integrate();
};

// Attractive soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVERA: public SoftParticleLangevin
{
public:
  SoftParticleNVERA() = default;
  SoftParticleNVERA(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void integrate();
};

// Fixed boundary soft particle NVE integrator child of SoftParticleLangevin
class SoftParticleNVEFixedBoundary: public SoftParticleLangevin
{
public:
  SoftParticleNVEFixedBoundary() = default;
  SoftParticleNVEFixedBoundary(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleLangevin2
class SoftParticleActiveLangevin: public SoftParticleLangevin2
{
public:
  SoftParticleActiveLangevin() = default;
  SoftParticleActiveLangevin(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleActiveLangevin
class SoftParticleActiveFixedBoundary: public SoftParticleActiveLangevin
{
public:
  SoftParticleActiveFixedBoundary() = default;
  SoftParticleActiveFixedBoundary(SP2D * spPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator child of softParticleActiveLangevin
class SoftParticleActiveFixedSides: public SoftParticleActiveLangevin
{
public:
  SoftParticleActiveFixedSides() = default;
  SoftParticleActiveFixedSides(SP2D * spPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Active Langevin integrator with massive particles child of SoftParticleActiveLangevin
class SoftParticleActiveSubSet: public SoftParticleLangevinSubSet
{
public:
  SoftParticleActiveSubSet() = default;
  SoftParticleActiveSubSet(SP2D * spPtr, SimConfig config) : SoftParticleLangevinSubSet:: SoftParticleLangevinSubSet(spPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Active Langevin integrator with external field child of SoftParticleActiveLangevin
class SoftParticleActiveExtField: public SoftParticleActiveLangevin
{
public:
  SoftParticleActiveExtField() = default;
  SoftParticleActiveExtField(SP2D * spPtr, SimConfig config) : SoftParticleActiveLangevin:: SoftParticleActiveLangevin(spPtr, config){;}

  virtual void integrate();
};

#endif // SIMULATOR_H //
