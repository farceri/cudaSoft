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
  double noise = 0; // this is just a choice
  double gamma = 1; // this is just a choice
  double gamma1;
  double gamma2;
  long firstIndex = 10;
  long extForce = 0;
  double mass = 1;
  thrust::device_vector<double> d_rand;
  thrust::device_vector<double> d_rando;
  thrust::device_vector<double> d_velSum; // for computing velocity drift

  SimInterface() = default;
  SimInterface(SP2D * spPtr, SimConfig config):sp_(spPtr),config(config){}
  virtual ~SimInterface() = default;

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

// Soft particle Langevin integrator child of SoftParticleLangevin
class SoftParticleDrivenLangevin: public SoftParticleLangevin
{
public:
  SoftParticleDrivenLangevin() = default;
  SoftParticleDrivenLangevin(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Langevin integrator child of SoftParticleLangevin
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

  virtual void injectKineticEnergy();
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Brownian integrator child of SoftParticleLangevin
class SoftParticleBrownian: public SoftParticleLangevin
{
public:
  SoftParticleBrownian() = default;
  SoftParticleBrownian(SP2D * spPtr, SimConfig config) : SoftParticleLangevin:: SoftParticleLangevin(spPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Soft particle Brownian integrator with driving force child of SoftParticleBrownian
class SoftParticleDrivenBrownian: public SoftParticleBrownian
{
public:
  SoftParticleDrivenBrownian() = default;
  SoftParticleDrivenBrownian(SP2D * spPtr, SimConfig config) : SoftParticleBrownian:: SoftParticleBrownian(spPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

#endif // SIMULATOR_H //
