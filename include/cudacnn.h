#ifndef __CUDACNN_H__
#define __CUDACNN_H__

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <list>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <assert.h>

#include "cnnconfig.h"

#include "cnnlimits.h"
#include "exceptions.h"
#include "common.h"
#include "tensor.h"

#ifndef __CUDACC__
#ifdef HAVE_HDF5
#include "cpp/H5Cpp.h"

#include "hdf5_helper.h"

#endif //HAVE_HDF5
#endif //__CUDACC__

#include "transfer_functions.h"
#include "performance_functions.h"
#include "layer.hpp"
#include "clayer.h"
#include "clayer_cuda.h"
#include "flayer.h"
#include "flayer_cuda.h"
#include "player.h"
#include "player_cuda.h"

#ifdef HAVE_BOOST
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif //HAVE_BOOST

#include "conv_net.h"
#include "trainer.h"

#endif //__CUDACNN_H__