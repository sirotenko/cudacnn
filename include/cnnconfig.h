#ifndef _CNNCONFIG_H_
#define _CNNCONFIG_H_
#define  CUDACNN_BUILD_SHARED_LIB

/* Name of package */
#define  PACKAGE "cudacnn"

/* Define to the full name of this package. */
#define  PACKAGE_NAME "cudacnn"

/* Define to the full name and version of this package. */
#define  PACKAGE_STRING "cudacnn 1.0.0"

/* Version number of package */
#define  CUDACNN_VERSION "1.0.0"

/* Boost C++ library*/
#define HAVE_BOOST

/* NVidia Cuda Runtime API*/
#define HAVE_CUDA

/* HDF5 Library*/
/* #undef HAVE_HDF5 */

/* version of cnn format in hdf5 file */
#define __CNN_FILE_VERSION 2
/* Google Testing Framework Library*/
/* #undef HAVE_GTEST */
#endif //CNNCONFIG_H
