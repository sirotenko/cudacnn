#Based on OpenCV cmake script
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(VERSION 2.8)
ENDIF(COMMAND CMAKE_POLICY)

# CUDA accelerated Convolutional neutal network library 
project(cudacnn)
#Script debug
#SET( CMAKE_VERBOSE_MAKEFILE on )
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

if(UNIX)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /lib /usr/lib)
endif()

if (NOT CMAKE_INSTALL_PREFIX)
    if (WIN32)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE INTERNAL "" FORCE)
    elseif()
        set(CMAKE_INSTALL_PREFIX "/usr" CACHE INTERNAL "" FORCE)
    endif()
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration" FORCE)
set(WITH_CUDA ON CACHE BOOL "Build cuda version of library")
set(WITH_HDF5 OFF CACHE BOOL "HDF5 library to save and load neural networks")


if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

# ----------------------------------------------------------------------------
#  Current version number:
# ----------------------------------------------------------------------------

set(CUDACNN_VERSION "1.0.0")

string(REGEX MATCHALL "[0-9]" CUDACNN_VERSION_PARTS "${CUDACNN_VERSION}")

list(GET CUDACNN_VERSION_PARTS 0 CUDACNN_VERSION_MAJOR)
list(GET CUDACNN_VERSION_PARTS 1 CUDACNN_VERSION_MINOR)
list(GET CUDACNN_VERSION_PARTS 2 CUDACNN_VERSION_PATCH)

set(CUDACNN_SOVERSION "${CUDACNN_VERSION_MAJOR}.${CUDACNN_VERSION_MINOR}")

if(WIN32)
    # Postfix of DLLs:
    set(CUDACNN_DLLVERSION "${CUDACNN_VERSION_MAJOR}${CUDACNN_VERSION_MINOR}${CUDACNN_VERSION_PATCH}")
    set(CUDACNN_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    set(CUDACNN_DLLVERSION "")
    set(CUDACNN_DEBUG_POSTFIX)
endif()

# ----------------------------------------------------------------------------
#  Variables for cnnconfig.h.cmake
# ----------------------------------------------------------------------------
set(PACKAGE "cudacnn")
set(PACKAGE_NAME "cudacnn")
set(PACKAGE_STRING "${PACKAGE} ${CUDACNN_VERSION}")

# Build/install (or not) some apps:
# ===================================================
set(BUILD_MEX_FILES ON CACHE BOOL "Build Matlab mex files")

# Build tests:
# ===================================================
set(BUILD_TESTS OFF CACHE BOOL "Build tests")

# Build tools:
# ===================================================
set(BUILD_TOOLS OFF CACHE BOOL "Build tools")


include(cudacnnPCHSupport.cmake REQUIRED)

#if(MSVC)
#   set(DEFAULT_ENABLE_OPENMP ON)
#else()
#   set(DEFAULT_ENABLE_OPENMP OFF)
#endif()
#set(ENABLE_OPENMP ${DEFAULT_ENABLE_OPENMP} CACHE BOOL "")

#Taken from eblearn and slightly changed
# determine machine architecture
################################################################################
IF (APPLE) # MAC OS
  SET (MAC true)
  SET (OS_NAME "Mac")
ELSE (APPLE)
  IF("${CMAKE_SYSTEM}" MATCHES "Linux")
    SET (LINUX true)
    SET (OS_NAME "Linux")
  ELSE ("${CMAKE_SYSTEM}" MATCHES "Linux")
    SET (WINDOWS true)
    SET (OS_NAME "Windows")
  ENDIF("${CMAKE_SYSTEM}" MATCHES "Linux")
ENDIF (APPLE)

SET(BITSIZE "32")
SET(64BIT false) # default is 32 bits
IF (APPLE OR LINUX)
  EXEC_PROGRAM("uname -m" OUTPUT_VARIABLE ARCH_NAME)
  STRING(COMPARE EQUAL ${ARCH_NAME} "x86_64" 64BIT)
ELSE (APPLE OR LINUX)
  STRING(COMPARE EQUAL $ENV{PROCESSOR_ARCHITEW6432} "AMD64" 64BIT)
  #STRING(COMPARE EQUAL $ENV{PROCESSOR_ARCHITECTURE} "AMD64" 64BIT)
ENDIF (APPLE OR LINUX)
IF (64BIT)
  SET(BITSIZE "64")
ENDIF (64BIT)

MESSAGE(STATUS "Target OS is ${OS_NAME} (${BITSIZE} bits)")

# ----------------------------------------------------------------------------
# Detect GNU version:
# ----------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    set(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*)
        # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(X86_64 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(X86_64 1)
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        set(X86 1)
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()

    if(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        set(USE_O3 ON CACHE BOOL "Enable -O3")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math")
        set(ENABLE_SSE ON CACHE BOOL "Enable SSE instructions")
        set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 instructions")
        set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 instructions")
        set(ENABLE_SSSE3 OFF CACHE BOOL "Enable SSSE3 instructions")
        set(ENABLE_SSE41 OFF CACHE BOOL "Enable SSE4.1 instructions")
        set(ENABLE_SSE42 OFF CACHE BOOL "Enable SSE4.2 instructions")
    endif()
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# ----------------------------------------------------------------------------
# Default: dynamic libraries:
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
ELSE(CMAKE_COMPILER_IS_GNUCXX)
	SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
IF(BUILD_SHARED_LIBS)
    SET(CUDACNN_BUILD_SHARED_LIB 1) # For cnnconfig.h, etc.
ELSE(BUILD_SHARED_LIBS)
    SET(CUDACNN_BUILD_SHARED_LIB 0)
ENDIF(BUILD_SHARED_LIBS)

############################### Boost ################################
if(CMAKE_COMPILER_IS_GNUCXX)
	set(Boost_USE_STATIC_LIBS OFF)
	set(Boost_USE_MULTITHREADED ON) 
	set(Boost_USE_STATIC_RUNTIME OFF)
	find_package(Boost 1.42.0 REQUIRED)

	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIRS})
		set(HAVE_BOOST 1)
		message(STATUS "Boost found")			
	endif()
endif(CMAKE_COMPILER_IS_GNUCXX)
############################### CUDA ################################
if (WITH_CUDA)
	find_package(CUDA 4.2)
	if (CUDA_FOUND)		
		set(HAVE_CUDA 1)
		message(STATUS "CUDA detected: " ${CUDA_VERSION})
		
		set(CUDA_COMPUTE_CAPABILITIES " 2.0 " CACHE STRING "Add or remove compute capability")
		set(CUDA_NVCC_FLAGS_ARCH ${CUDA_COMPUTE_CAPABILITIES})

		set(CUDA_NVCC_FLAGS_NUM "")

		while(NOT ${CUDA_NVCC_FLAGS_ARCH} STREQUAL "")
		    string(REGEX MATCH "[0-9]+.[0-9]+" RESULT_NUM ${CUDA_NVCC_FLAGS_ARCH})
		    string(REGEX MATCHALL "[0-9]" RESULT_STR ${RESULT_NUM})
		    string(REGEX REPLACE ";" "\ " RESULT ${RESULT_STR})
		    list(APPEND CUDA_NVCC_FLAGS_NUM ${RESULT})
		    string(REGEX REPLACE "${RESULT_NUM}" "\ " CUDA_NVCC_FLAGS_ARCH_STR ${CUDA_NVCC_FLAGS_ARCH})
		    string(STRIP ${CUDA_NVCC_FLAGS_ARCH_STR} CUDA_NVCC_FLAGS_ARCH)
		endwhile()

		set (CUDACNN_CUDA_CC "")
		set (loop_var "")
		foreach( loop_var IN LISTS CUDA_NVCC_FLAGS_NUM)
		    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${loop_var},code=sm_${loop_var})
		    set (CUDACNN_CUDA_CC ${CUDACNN_CUDA_CC} -gencode arch=compute_${loop_var},code=sm_${loop_var})
		endforeach()

	endif()
endif()
############################### Matlab ################################
if(BUILD_MEX_FILES)
	find_package(Matlab REQUIRED)
	if (MATLAB_FOUND)
		set(HAVE_MATLAB 1)
		message(STATUS "Matlab found")		
	else()
		message(STATUS "Matlab not found. Please specify the paths to Matlab libs and headers")
		set(MATLAB_INCLUDE_DIR "" CACHE PATH "MATLAB: include path for mex.h, engine.h, etc")
		set(MATLAB_MEX_LIBRARY "" CACHE FILEPATH "MATLAB: path to libmex.lib")		
		set(MATLAB_MX_LIBRARY "" CACHE FILEPATH "MATLAB: path to libmx.lib")		
		set(MATLAB_ENG_LIBRARY "" CACHE FILEPATH "MATLAB: path to libeng.lib")		
		set(MATLAB_MEXFILE_EXT "" CACHE STRING "MATLAB: extension of mex-files")		
		SET(MATLAB_LIBRARIES
		  ${MATLAB_MEX_LIBRARY}
		  ${MATLAB_MX_LIBRARY}
		  ${MATLAB_ENG_LIBRARY}
		)
	endif()		
endif()
############################### HDF5 ################################
if(WITH_HDF5)
	set(HDF5_FIND_COMPONENTS "C;CXX")
	find_package(HDF5 REQUIRED ${HDF5_FIND_COMPONENTS})
	if (HDF5_FOUND)
		set(HAVE_HDF5 1)
		message(STATUS "HDF5 found")		
	else()
		message(STATUS "HDF5 not found. Please specify the paths to HDF5 libs and headers")
	endif()		
	#In windows FindHDF5 sometimes give strange results, so need to check this and fix manually if needed
	set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} CACHE PATH "HDF5: include path" FORCE)
	set(HDF5_LIBRARIES ${HDF5_LIBRARIES} CACHE PATH "HDF5: hdf5 lib path" FORCE)	
	#message(STATUS "HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIRS}")
	#message(STATUS "HDF5_LIBRARIES = ${HDF5_LIBRARIES}")
	include_directories(${HDF5_INCLUDE_DIRS})	
endif()
############################### GTest ################################
if(BUILD_TESTS)
	#set(GTEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/gtest)
	#find_package(GTest REQUIRED)
	#if (GTEST_FOUND)
#		set(HAVE_GTEST 1)
#		message(STATUS "GTest found")		
#	endif()		
#Instead of using find_package the recommendation is to include gtest sources and build 
#https://groups.google.com/forum/?fromgroups=#!topic/googletestframework/zHy-rrKz_Ho
	add_subdirectory(3rdParty/gtest)
	enable_testing(true)
endif()

############################### GTest ################################
if(BUILD_TOOLS)
	add_subdirectory(tools)
endif()


add_definitions(-DHAVE_CONFIG_H)

set(CUDACNN_CONFIG_FILE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Where to create the platform-dependant cnnconfig.h")

message(STATUS "Parsing 'cnnconfig.h.cmake'")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cnnconfig.h.cmake" "${CUDACNN_CONFIG_FILE_INCLUDE_DIR}/cnnconfig.h")

# ---------------------------------------------------------------------------
# The C+//0 include & link directories:
# ---------------------------------------------------------------------------
include_directories("."
	"${CUDACNN_CONFIG_FILE_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
# ----------------------------------------------------------------------------
#           Set the maximum level of warnings:
# ----------------------------------------------------------------------------

# Should be set to true for development
set(CUDACNN_WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")

set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")

if(MSVC)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
    # 64-bit portability warnings, in MSVC8
    if(MSVC80)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Wp64")
    endif()
    #if(MSVC90)
    #    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _BIND_TO_CURRENT_CRT_VERSION=1 /D _BIND_TO_CURRENT_VCLIBS_VERSION=1")
    #endif()

    set(EXTRA_EXE_LINKER_FLAGS_RELEASE "${EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")

    if(ENABLE_OPENMP)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /openmp")
    endif()

    # Remove unreferenced functions: function level linking
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Gy")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Zi")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)

    # High level of warnings.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    if(CUDACNN_WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif()

    if(X86)
        if(NOT MINGW64)
            if(NOT X86_64)
                if(NOT APPLE)
                    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
                endif()
            endif()
        endif()
    endif()

    # Other optimizations
    if(USE_OMIT_FRAME_POINTER)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    if(USE_O2)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    endif()
    if(USE_FAST_MATH)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(ENABLE_POWERPC)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(ENABLE_SSE)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse")
    endif()
    if(ENABLE_SSE2)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse2")
    endif()
    # SSE3 and further should be disabled under MingW because it generates compiler errors
    if(NOT MINGW)
        if(ENABLE_SSE3)
            set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse3")
        endif()
        
#        if(${CMAKE_CUDACNN_GCC_VERSION_NUM} GREATER 402)
#            set(HAVE_GCC43_OR_NEWER 1)
#        endif()
#        if(${CMAKE_CUDACNN_GCC_VERSION_NUM} GREATER 401)
#            set(HAVE_GCC42_OR_NEWER 1)
#        endif()
        
#        if(HAVE_GCC43_OR_NEWER OR APPLE)
            if(ENABLE_SSSE3)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
            endif()
 #           if(HAVE_GCC42_OR_NEWER)
                if(ENABLE_SSE41)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
                endif()
                if(ENABLE_SSE42)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.2")
                endif()
  #          endif()
   #     endif()
    endif()

    if(X86 OR X86_64)
        if(NOT APPLE)
            if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
            endif()
        endif()
    endif()

	# Remove unreferenced functions: function level linking
	if(NOT APPLE)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
	endif()

    # Parallel mode
    if(ENABLE_OPENMP)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -D_GLIBCXX_PARALLEL -fopenmp")
        set(CUDACNN_LINKER_LIBS ${CUDACNN_LINKER_LIBS} gomp)
    endif()

    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -ggdb3 -DDEBUG -D_DEBUG")
endif()

# Extra link libs if the user selects building static libs:
IF(NOT BUILD_SHARED_LIBS)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CUDACNN_LINKER_LIBS ${CUDACNN_LINKER_LIBS} stdc++)
    endif()
endif()

# Add user supplied extra options (optimization, etc...)
# ==========================================================
#set(CUDACNN_EXTRA_C_FLAGS "" CACHE STRING "Extra compiler options")
#set(CUDACNN_EXTRA_C_FLAGS_RELEASE "" CACHE STRING "Extra compiler options for Release build")
#set(CUDACNN_EXTRA_C_FLAGS_DEBUG "" CACHE STRING "Extra compiler options for Debug build")
#set(CUDACNN_EXTRA_EXE_LINKER_FLAGS "" CACHE STRING "Extra linker flags" FORCE)
#set(CUDACNN_EXTRA_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "Extra linker flags for Release build" FORCE)
#set(CUDACNN_EXTRA_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Extra linker flags for Debug build" FORCE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${CUDACNN_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} ${CUDACNN_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${CUDACNN_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${CUDACNN_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${CUDACNN_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${CUDACNN_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS} ${CUDACNN_EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE} ${CUDACNN_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG} ${CUDACNN_EXTRA_EXE_LINKER_FLAGS_DEBUG}")

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

if("${CMAKE_CONFIGURE_LDFLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_CONFIGURE_LDFLAGS}")
endif("${CMAKE_CONFIGURE_LDFLAGS}")

# ----------------------------------------------------------------------------
#                       PROCESS SUBDIRECTORIES:
# ----------------------------------------------------------------------------
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

#-----------------------------------
# Subdirectories:
#-----------------------------------
add_subdirectory(libcudacnn)
if(BUILD_MEX_FILES AND MATLAB_FOUND)
	add_subdirectory(mexFunction)
endif()
if(BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
#   Sumary:
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "General configuration for cudacnn ${CUDACNN_VERSION} =====================================")
message(STATUS "")
message(STATUS "    Built as dynamic libs?:    ${BUILD_SHARED_LIBS}")
message(STATUS "    Compiler:                  ${CMAKE_COMPILER}")
message(STATUS "    C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "    C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if(WIN32)
message(STATUS "    Linker flags (Release):    ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message(STATUS "    Linker flags (Debug):      ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
message(STATUS "    Linker flags (Release):    ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
message(STATUS "    Linker flags (Debug):      ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()

message(STATUS "")
message(STATUS "  Interfaces: ")
if (HAVE_CUDA)
message(STATUS "    Use Cuda:                  YES")
else()
message(STATUS "    Use Cuda:                  No")
endif()

if(WITH_HDF5 AND HDF5_FOUND)
message(STATUS "    Use Hdf5:                  YES")
else()	
message(STATUS "    Use Hdf5:                  NO")
endif()

if(BUILD_MEX_FILES)
message(STATUS "    Build mex files:           YES")
else()	
message(STATUS "    Build mex files:           NO")
endif()

if(MATLAB_FOUND)
message(STATUS "    Matlab found:              YES")
else()	
message(STATUS "    Matlab found:              NO")
endif()
if(Boost_FOUND)
message(STATUS "    Boost found:               YES")
else()	
message(STATUS "    Boost found:               NO")
endif()


message(STATUS "")
message(STATUS "    Install path:              ${CMAKE_INSTALL_PREFIX}")
