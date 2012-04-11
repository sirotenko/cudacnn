#Taken from OpenCV and slightly changed
cmake_minimum_required(VERSION 2.8.6 FATAL_ERROR)

IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(VERSION 2.8)
  #CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

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
set(USE_DOUBLE OFF CACHE BOOL "Use double precision instead float")
set(WITH_CUDA ON CACHE BOOL "Build cuda version of library")
set(WITH_HDF5 OFF CACHE BOOL "HDF5 library to save and load neural networks")

project(cudacnn)

if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

#Some of CMAKE Find scripts not working propertly
# set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_files 
  # ${CMAKE_MODULE_PATH})

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
# Build static or dynamic libs?
# ----------------------------------------------------------------------------
# Default: dynamic libraries:
SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
IF(BUILD_SHARED_LIBS)
    SET(CUDACNN_BUILD_SHARED_LIB 1) # For cnnconfig.h, etc.
ELSE(BUILD_SHARED_LIBS)
    SET(CUDACNN_BUILD_SHARED_LIB 0)
ENDIF(BUILD_SHARED_LIBS)

# ----------------------------------------------------------------------------
#  Variables for cnnconfig.h.cmake
# ----------------------------------------------------------------------------
set(PACKAGE "cudacnn")
set(PACKAGE_NAME "cudacnn")
set(PACKAGE_STRING "${PACKAGE} ${CUDACNN_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE}")
set(PACKAGE_VERSION "${CUDACNN_VERSION}")

# Build/install (or not) some apps:
# ===================================================
set(BUILD_MEX_FILES ON CACHE BOOL "Build Matlab mex files")
set(BUILD_C_EXAMPLES OFF CACHE BOOL "Build C examples")

# Build tests:
# ===================================================
set(BUILD_TESTS OFF CACHE BOOL "Build tests")

include(OpenCVPCHSupport.cmake REQUIRED)

if(MSVC)
   set(DEFAULT_ENABLE_OPENMP ON)
else()
   set(DEFAULT_ENABLE_OPENMP OFF)
endif()
set(ENABLE_OPENMP ${DEFAULT_ENABLE_OPENMP} CACHE BOOL "")

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

############################### CUDA ################################
if (WITH_CUDA)
	find_package(CUDA 3.2)
	if (CUDA_FOUND)		
		set(HAVE_CUDA 1)
		message(STATUS "CUDA detected: " ${CUDA_VERSION})
		
		set(CUDA_COMPUTE_CAPABILITIES " 1.1 1.2 1.3 2.0 " CACHE STRING "Add or remove compute capability")
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
		    set (CUDACNN_CUDA_CC ${CUDACNN_CUDA_CC}  -gencode arch=compute_${loop_var},code=sm_${loop_var})
		endforeach()

	endif()
endif()
############################### GTest ################################
if(BUILD_TESTS)
	find_package(GTest REQUIRED)
	if (GTEST_FOUND)
		set(HAVE_GTEST 1)
		message(STATUS "GTest found")		
	endif()		
endif()
############################### Matlab ################################
if(BUILD_MEX_FILES)
	find_package(Matlab REQUIRED)
	if (MATLAB_FOUND)
		set(HAVE_MATLAB 1)
		message(STATUS "Matlab found")		
	endif()		
endif()
############################### Matlab ################################
if(WITH_HDF5)
	set(HDF5_FIND_COMPONENTS "C;CXX")
	find_package(HDF5 REQUIRED ${HDF5_FIND_COMPONENTS})
	if (HDF5_FOUND)
		set(HAVE_HDF5 1)
		message(STATUS "HDF5 found")		
		# message(STATUS ${HDF5_INCLUDE_DIRS})		
		# message(STATUS ${HDF5_LIBRARIES})		
		# message(STATUS ${HDF5_LIBRARY_DIRS})
		# message(STATUS ${HDF5_CXX_LIBRARIES})
		# message(STATUS ${HDF5_LIBRARIES_RELEASE})
	endif()		
endif()


# ----------------------------------------------------------------------------
#                   UPDATE CONFIG FILES & SCRIPTS:
#
#  CONFIGURE_FILE(InputFile OutputFile [COPYONLY] [ESCAPE_QUOTES] [@ONLY])
# If @ONLY is specified, only variables of the form @VAR@ will be
#  replaces and ${VAR} will be ignored.
#
#  A directory will be created for each platform so the "cnnconfig.h" file is
#   not overwritten if cmake generates code in the same path.
# ----------------------------------------------------------------------------
add_definitions(-DHAVE_CONFIG_H)

set(CUDACNN_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant cnnconfig.h")

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
        
        if(${CMAKE_CUDACNN_GCC_VERSION_NUM} GREATER 402)
            set(HAVE_GCC43_OR_NEWER 1)
        endif()
        if(${CMAKE_CUDACNN_GCC_VERSION_NUM} GREATER 401)
            set(HAVE_GCC42_OR_NEWER 1)
        endif()
        
        if(HAVE_GCC43_OR_NEWER OR APPLE)
            if(ENABLE_SSSE3)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
            endif()
            if(HAVE_GCC42_OR_NEWER)
                if(ENABLE_SSE41)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
                endif()
                if(ENABLE_SSE42)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.2")
                endif()
            endif()
        endif()
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
set(CUDACNN_EXTRA_C_FLAGS "" CACHE STRING "Extra compiler options")
set(CUDACNN_EXTRA_C_FLAGS_RELEASE "" CACHE STRING "Extra compiler options for Release build")
set(CUDACNN_EXTRA_C_FLAGS_DEBUG "" CACHE STRING "Extra compiler options for Debug build")
set(CUDACNN_EXTRA_EXE_LINKER_FLAGS "" CACHE STRING "Extra linker flags" FORCE)
set(CUDACNN_EXTRA_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "Extra linker flags for Release build" FORCE)
set(CUDACNN_EXTRA_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Extra linker flags for Debug build" FORCE)

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
#add_subdirectory(include)
add_subdirectory(libcudacnn)
if(BUILD_MEX_FILES)
	add_subdirectory(mexFunction)
endif()
if(BUILD_C_EXAMPLES)
	add_subdirectory(samples)
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
message(STATUS "    Hdf5:	                   YES")
else()	
message(STATUS "    Hdf5:                 	   NO")
endif()
message(STATUS "")
message(STATUS "    Install path:              ${CMAKE_INSTALL_PREFIX}")
