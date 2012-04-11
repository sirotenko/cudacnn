/* OpenCV compiled as static or dynamic libs */
#cmakedefine  CUDACNN_BUILD_SHARED_LIB

/* Name of package */
#define  PACKAGE "${PACKAGE}"

/* Define to the full name of this package. */
#define  PACKAGE_NAME "${PACKAGE_NAME}"

/* Define to the full name and version of this package. */
#define  PACKAGE_STRING "${PACKAGE_STRING}"

/* Define to the one symbol short name of this package. */
#define  PACKAGE_TARNAME "${PACKAGE_TARNAME}"

/* Define to the version of this package. */
#define  PACKAGE_VERSION "${PACKAGE_VERSION}"

/* Version number of package */
#define  VERSION "${PACKAGE_VERSION}"

/* NVidia Cuda Runtime API*/
#cmakedefine HAVE_CUDA

/* HDF5 Library*/
#cmakedefine HAVE_HDF5

/* Google Testing Framework Library*/
#cmakedefine HAVE_GTEST
