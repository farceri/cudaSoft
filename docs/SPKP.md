# How to produce and evolve configurations of Self-Propelled Kuramoto Particles

Self-Propelled Kuramoto Particles (SPKPs) are active particles subject to a driving force given by the same of self-propulsion and Kuramoto-type neighbor alignment. This codebase generates and evolves particle configurations in circular confinement to study their collective behavior.

The workflow consists of three main steps:

Initial packing generation (compressPacking.cpp): produces disordered configurations at a target density using purely repulsive interactions and smooth boundaries.
Active dynamics (runKuramoto.cpp): evolves particles with driven Brownian dynamics, including persistence and velocity alignment interactions between.
Boundary condition switch (runWall.cpp, optional): replaces smooth confinement with rough or pinned boundaries to model wall-induced effects.

Particles interact via short-range repulsive potentials (WCA, harmonic, or LJ). All simulations are performed in 2D circular geometry, and configurations are stored as particle-resolved data (positions, velocities, radii, and driving angles). The code is designed to systematically explore the role of density, persistence time, and alignment in non-equilibrium phase behavior.

# Driving force parameters (runKuramoto.cpp, runWall.cpp)
- Jk, Kuramoto coupling strength. Controls alignment between particle driving directions.
- taup, persistence time of the self-propulsion direction.
- alignment_type: vel, Kuramoto alignment based on particle velocities.


# Quick start
# 1. Generate initial packing
./compressPacking /data/ 1e-4 1e-1 1024 2 1 0 0
# 2. Run SPKP dynamics
./runKuramoto /data/0.014/ 5e-5 1 1e3 1e2 1e7 0 1024 wca reflect vel 0
# 3. (Optional) Add rough boundary
./runWall /data/0.014/reflect/ 5e-5 1 1e3 1e2 1e7 0 1024 wca rough vel 0.99 1


# Data structure
Particle configurations are saved in a directory containing the following default files:
- boxSize.dat, box radius
- particlePos.dat, particle positions
- particleRad.dat, particle radii
- particleVel.dat, particle velocities
- particleAngles.dat, angles of the driving force acting on each particle
- params.dat, general info on the state of the sample (number of particles, number of dimensions, time step, density, etc.)
- dynParams.dat, equation of motion parameters (damping coefficient, persistence time, Kuramoto parameters)


# Boundary conditions (enumType in cudaSoft)
- smooth boundary (reflect): the sign of the radial particle velocity is flipped when a particles is closer than its radius to the wall.
- rough boundary (rough): the boundary is a rigid ring polymer with Nm monomers at contact with each other. Weeks-Chandler-Andersen or Inverse Power Law potentials are used to compute interactions between particles and boundary monomers.
- pinned boundary (rigid): pinned version of rough boundary where the polymer (wall) can rotate around its center due to particle interactions. The wall angular coordinate is evolved using Brownian dynamics.


# Compressing sweep until desired density
Configurations at different density are produced via the compressPacking.cpp script.
Particles are produced with 20% size polydispersity by extracting their radii from a log-normal distribution.
The scripts runs with smooth boundary conditions only and the interparticle potential is WCA.
Configurations are saved in directories named /dirPath/reflect/<currentDensity>/.

Example usage within circular geometry:
./compressPacking /dirPath/ <timestep>=1e-04 <Tinject>=1e-01 <numParticles>=32768 <dim>=2 <boxRadius>=1 0 0


# Run driven Brownian dynamics
Use runKuramoto.cpp to simulate the time evolution of SPKPs. The script typically reads configurations produced with compressPacking.cpp.
The interparticle potential can be choosen between: harmonic, WCA, and LJ. Refer to SP2D.h for more.
The integrator type can be choosen between: NVE, Langevin, Brownian, and driven Brownian (default 0).

Example usage within smooth boundary conditions:
./runKuramoto /dirPath/0.014/ <timstep>=5e-05 <Jk>=1 <taup>=1e03 <damping>=1e02 <numSteps>=1e07 <initialStep>=0 <numParticles>=1024 <interparticle_potential>=wca <boundary_type>=reflect <alignment_type>=vel <integrator_type>=0


# Evolve configurations in different boundary conditions
Use runWall.cpp to evolve configurations previously produced in smooth boundary conditions using runKuramoto.cpp.
The script runs on driven Brownian dynamics only. Additional input parameters with respect to runKuramoto are:
- scale, a scalar number used to shrink radial particle coordinates to avoid big jumps in potential energies when switching from smooth to rough boundaries.
- roughness, ratio between particle radius and boundary monomer diameter. If it equal to 1, boundary monomers are half as big as the average particle.

Example usage within rough boundary conditions:
./runWall /dirPath/0.014/reflect/ <timstep>=5e-05 <Jk>=1 <taup>=1e03 <damping>=1e02 <numSteps>=1e07 <initialStep>=0 <numParticles>=1024 <interparticle_potential>=wca <boundary_type>=reflect <alignment_type>=vel <scale>=0.99 <roughness>=1


Please contact arceri.fra at gmail.com for more info and clarifications.

Example of SPKPs in smooth boundary conditions
<img width="1280" height="960" alt="vel-phi05-j3-tp1e03" src="https://github.com/user-attachments/assets/6ff7a2cb-0b48-4293-8022-e874eb840880" />

