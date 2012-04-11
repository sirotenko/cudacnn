################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../tests/cudacnn_test.cpp 

OBJS += \
./tests/cudacnn_test.o 


# Each subdirectory must supply rules for building sources it contributes
tests/%.o: ../tests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: C++ Compiler'
	g++ -I"/home/sirotenko/Projects/CudaCnn/include" -I$(CUDAHOME)/include -I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc -I$(MATLAB)/extern/include -O2 -g -Wall -c -fmessage-length=0 -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


