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
  SimConfig() = default;
  SimConfig(double Tin, double Dr, double driving):Tinject(Tin), Dr(Dr), driving(driving){}
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
  long extForce = 0;
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
class SoftParticleLangevinExtField: public SoftParticleLangevin2
{
public:
  SoftParticleLangevinExtField() = default;
  SoftParticleLangevinExtField(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with perturbation child of softParticleLangevin2
class SoftParticleLangevinPerturb: public SoftParticleLangevin2
{
public:
  SoftParticleLangevinPerturb() = default;
  SoftParticleLangevinPerturb(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void integrate();
};

// Soft particle Langevin integrator with fluid flow child of softParticleLangevin2
class SoftParticleLangevinFlow: public SoftParticleLangevin2
{
public:
  SoftParticleLangevinFlow() = default;
  SoftParticleLangevinFlow(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Soft particle integrator with fluid flow child of softParticleLangevin2
class SoftParticleFlow: public SoftParticleLangevin2
{
public:
  SoftParticleFlow() = default;
  SoftParticleFlow(SP2D * spPtr, SimConfig config) : SoftParticleLangevin2:: SoftParticleLangevin2(spPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
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

// Soft particle NVE integrator with velocity rescale child of SoftParticleLangevin
class SoftParticleNVERescale: public SoftParticleLangevin
{
public:
  SoftParticleNVERescale() = default;
  SoftParticleNVERescale(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void integrate();
};

// Soft particle NVE integrator with double velocity rescale child of SoftParticleLangevin
class SoftParticleNVEDoubleRescale: public SoftParticleLangevin
{
public:
  SoftParticleNVEDoubleRescale() = default;
  SoftParticleNVEDoubleRescale(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void integrate();
};

// Soft particle Nose-Hoover integrator child of SoftParticleLangevin
class SoftParticleNoseHoover: public SoftParticleLangevin
{
public:
  SoftParticleNoseHoover() = default;
  SoftParticleNoseHoover(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle double temperature Nose-Hoover integrator child of SoftParticleLangevin
class SoftParticleDoubleNoseHoover: public SoftParticleLangevin
{
public:
  SoftParticleDoubleNoseHoover() = default;
  SoftParticleDoubleNoseHoover(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
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
