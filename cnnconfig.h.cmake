#ifndef _CNNCONFIG_H_
#define _CNNCONFIG_H_
#cmakedefine  CUDACNN_BUILD_SHARED_LIB

/* Name of package */
#define  PACKAGE "${PACKAGE}"

/* Define to the full name of this package. */
#define  PACKAGE_NAME "${PACKAGE_NAME}"

/* Define to the full name and version of this package. */
#define  PACKAGE_STRING "${PACKAGE_STRING}"

/* Version number of package */
#define  CUDACNN_VERSION "${CUDACNN_VERSION}"

/* Boost C++ library*/
#cmakedefine HAVE_BOOST

/* NVidia Cuda Runtime API*/
#cmakedefine HAVE_CUDA

/* HDF5 Library*/
#cmakedefine HAVE_HDF5

/* version of cnn format in hdf5 file */
#define __CNN_FILE_VERSION 2
/* Google Testing Framework Library*/
#cmakedefine HAVE_GTEST
#endif //CNNCONFIG_H
