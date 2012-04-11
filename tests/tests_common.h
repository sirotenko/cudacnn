#ifndef _TESTS_COMMON_H
#define _TESTS_COMMON_H
#include <limits.h>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include "gtest/gtest.h"
#include "common.h"
#include "tensor.h"

//#include <mex.h>
//#include <mat.h>
#include "exceptions.h"

#include "layer.hpp"
#include "clayer_cuda.h"
#include "player_cuda.h"
#include "flayer_cuda.h"

#include "clayer.h"
#include "player.h"
#include "flayer.h"

#include "transfer_functions.h"
#include "performance_functions.h"
#include "conv_net.h"
//#include "matlab_import_export.h"

#define GPU_FLOAT_TOLLERANCE 0.00001

template <class T>
void ReadCSV(Tensor<T>* buff, std::string filename)
{
	unsigned i = 0;
	std::ifstream  data(filename.c_str());
	std::string line;
	while(std::getline(data,line))
	{
		if (i > buff->num_elements()) {
			throw std::runtime_error("Buffer size is less than number of elements in CSV-file");
		}
		std::stringstream  line_stream(line);
		std::string        cell;
		while(std::getline(line_stream,cell,','))
		{
			std::stringstream  cell_stream(cell);
			T val;
			cell_stream>>val;
			buff->data()[i] = val;
			i++;
		}
	}
}



template <class T>
void AssertNearTensors(const Tensor<T>& host, const TensorGPU<T>& gpu, std::string message)
{
	//Check if both null
	if(host.data() == NULL && gpu.data() == NULL ) return;
	Tensor<T> gpu_host = gpu;
	ASSERT_EQ(host.num_elements(), gpu_host.num_elements())<<"Tensors have different number of elements";
	for (UINT i = 0; i < host.num_elements(); ++i) {
		ASSERT_NEAR(host[i],gpu_host[i], GPU_FLOAT_TOLLERANCE)<<message<<"Host and CUDA tensors have different values at i = "<<i;
	}
}

#endif