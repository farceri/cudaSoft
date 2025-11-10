###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
#CUDA_ROOT_DIR=/usr/local/cuda-12.2
CUDA_ROOT_DIR=/usr/lib/x86_64-linux-gnu
#CUDA_ROOT_DIR=/gpfs/loomis/apps/avx/software/CUDAcore/11.3.1

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=/usr/bin/g++-11
#CC=/gpfs/loomis/apps/avx/software/GCCcore/10.2.0/bin/g++
CC_FLAGS= -O3
CC_LIBS= -lstdc++fs

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=/usr/bin/nvcc
#NVCC=nvcc
NVCC_FLAGS= -O3 -Wno-deprecated-gpu-targets --expt-extended-lambda --expt-relaxed-constexpr -ccbin /usr/bin/g++-11 #-G
NVCC_LIBS=

LFLAGS= -lm -Wno-deprecated-gpu-targets

# CUDA library directory:
#CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
#CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:

#EXE = testDynamics
#EXE = testInteraction

# make packings
#EXE = jamPacking
#EXE = compressPacking

#EXE = measurePressure
#EXE = measureTemperature

# run dynamics
EXE = runNVE
#EXE = runNVE2LJ
#EXE = runNH
#EXE = runNH2LJ
#EXE = runNVT
#EXE = runNVT2LJ
#EXE = runActive
#EXE = runActive2LJ
#EXE = runDoubleNH2LJ
#EXE = runExternalField
#EXE = runNPT
#EXE = runABP
#EXE = runVicsek
#EXE = runWall

# mechanics
#EXE = linearShearFIRE
#EXE = linearExtendNVE
#EXE = linearExtendNVE2LJ
#EXE = linearExtendNH2LJ
#EXE = linearExtendNVT
#EXE = linearExtendNVT2LJ
#EXE = linearExtendActive
#EXE = linearExtendActive2LJ
#EXE = extendNVT
#EXE = extendActive
#EXE = linearShearNVT
#EXE = shearNVT

# hydrodynamics
#EXE = runFlow

# Object files:
OBJS = $(OBJ_DIR)/$(EXE).o $(OBJ_DIR)/SP2D.o $(OBJ_DIR)/FIRE.o $(OBJ_DIR)/Simulator.o

##########################################################

## Compile ##

# Compiler-specific flags:
GENCODE_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM86 = -gencode=arch=compute_86,code=\"sm_86,compute_86\"

GENCODE = $(GENCODE_SM86)

# Link C++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(GENCODE) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CC_LIBS)
#	$(NVCC) $(GENCODE) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/$(EXE).o : $(EXE).cpp
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.h
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cuh
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
