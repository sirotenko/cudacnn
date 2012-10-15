//Copyright (c) 2012, Mikhail Sirotenko <mihail.sirotenko@gmail.com>
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __CUDACNNMEX_PRECOMP_H__
#define __CUDACNNMEX_PRECOMP_H__

#include <string>
#include <sstream>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <list>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdint.h>

// Matlab includes
#include <mex.h>
#include <mat.h>
#include <matrix.h>

#include "cnnconfig.h"

#ifdef HAVE_BOOST
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif //HAVE_BOOST


#ifdef HAVE_CUDA
#include <cuda.h>
#endif

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
//#endif

#ifdef HAVE_HDF5
//TODO: Find out why it's different and fix if possible
#ifdef _WIN32
#include "cpp/H5Cpp.h"
#else
#include "H5Cpp.h"
#endif  //_WIN32

#include "hdf5_helper.h"

#endif //HAVE_HDF5

#include "conv_net.h"
#include "trainer.h"

#include "mexcnnconfig.h"

#include "matlab_tools.h"
#include "matlab_import_export.h"

#endif