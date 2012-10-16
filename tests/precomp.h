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

#ifndef __CUDACNNTESTS_PRECOMP_H__
#define __CUDACNNTESTS_PRECOMP_H__

#include "cnnconfig.h"

#include <limits.h>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <time.h>
#include <memory>
//#include <array>
#include <list>

#include "gtest/gtest.h"
#include "common.h"
#include "tensor.h"

#include "exceptions.h"


#ifdef HAVE_HDF5
//TODO: Find out why it's different and fix if possible
#ifdef _WIN32
#include "cpp/H5Cpp.h"
#else
#include "H5Cpp.h"
#endif  //_WIN32

#include "hdf5_helper.h"

#endif //HAVE_HDF5


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

#include "tests_common.h"




#endif