#ifndef __CUDACNN_PRECOMP_H__
#define __CUDACNN_PRECOMP_H__

#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <list>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstring>


#include "cnnconfig.h"

#ifdef HAVE_BOOST
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif //HAVE_BOOST


#ifdef HAVE_CUDA
    #include "cuda_runtime_api.h"

#define CUDART_MINIMUM_REQUIRED_VERSION 3020

#if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
    #error "Insufficient Cuda Runtime library version, please update it."
#endif

#endif /* HAVE_CUDA */

#include "cnnlimits.h"
#include "common.h"
#include "assert.h"
#include "exceptions.h"
#include "tensor.h"

#ifdef HAVE_CUDA
#include "tensor_cuda.h"
#include "utils.cuh"
#endif

#include "transfer_functions.h"
#include "performance_functions.h"

#include "layer.hpp"
#include "clayer.h"
#include "flayer.h"
#include "player.h"

//#ifdef HAVE_CUDA
#include "clayer_cuda.h"
#include "flayer_cuda.h"
#include "player_cuda.h"

#ifndef __CUDACC__
#ifdef HAVE_HDF5
//TODO: Find out why it's different and fix if possible
//#ifdef _WIN32
#include "cpp/H5Cpp.h"
//#else
//#include "H5Cpp.h"
//#endif  //_WIN32

#include "hdf5_helper.h"

#endif //HAVE_HDF5
#endif //__CUDACC__

#endif /* __CUDACNN_PRECOMP_H__ */
