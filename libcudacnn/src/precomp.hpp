#ifndef __CUDACNN_PRECOMP_H__
#define __CUDACNN_PRECOMP_H__

#ifdef HAVE_CONFIG_H
#include "cnnconfig.h"
#endif


#include <string>
#include <sstream>
#include <exception>

#include "common.h"
#include "assert.h"
#include "tensor.h"

#include "exceptions.h"
#include "transfer_functions.h"
#include "layer.hpp"
#include "clayer.h"
#include "player.h"
#include "flayer.h"

//#if defined(HAVE_HDF5)
	#include "cpp/H5Cpp.h"
	#include "hdf5_helper.h"
//#endif

//#if defined(HAVE_CUDA)
    #include "cuda_runtime_api.h"

#define CUDART_MINIMUM_REQUIRED_VERSION 3020

#if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
    #error "Insufficient Cuda Runtime library version, please update it."
#endif

//#endif /* HAVE_CUDA */
#endif /* __CUDACNN_PRECOMP_H__ */
