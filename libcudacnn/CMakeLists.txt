project(cudacnnlib)

set(the_target "cudacnn")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_cuda "src/cuda/*.cu*")

source_group("Src\\" FILES ${lib_srcs})
source_group("Src\\Cuda" FILES ${lib_cuda})
file(GLOB lib_hdrs "../include/*.h*")
source_group("Include" FILES ${lib_hdrs})

if (HAVE_CUDA)		
	include_directories(${CUDA_INCLUDE_DIRS})	
	
	if (UNIX OR APPLE)
		set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
	endif()

	string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
	string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4211 /wd4201 /wd4100 /wd4505 /wd4408")

		string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
		string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
		string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	endif()

	CUDA_COMPILE(cuda_objs ${lib_cuda})
endif()

if(CUDACNN_BUILD_SHARED_LIB)
	#For Visual studio CUDA creates build rules, so we need to add cuda sources also
	if(${CMAKE_GENERATOR} MATCHES "Visual*") 
		add_library(${the_target} SHARED ${lib_srcs} ${lib_cuda} ${lib_hdrs} ${cuda_objs})
	else()
		add_library(${the_target} SHARED ${lib_srcs} ${lib_hdrs} ${cuda_objs})
	endif()
else(CUDACNN_BUILD_SHARED_LIB)
	if(${CMAKE_GENERATOR} MATCHES "Visual*") 
		add_library(${the_target} STATIC ${lib_srcs} ${lib_cuda} ${lib_hdrs} ${cuda_objs})
	else()
		add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs} ${cuda_objs})
	endif()
endif(CUDACNN_BUILD_SHARED_LIB)

if(PCHSupport_FOUND)
	set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)
	if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
		if(${CMAKE_GENERATOR} MATCHES "Visual*")
			set(${the_target}_pch "src/precomp.cpp")
		endif()            
		add_native_precompiled_header(${the_target} ${pch_header})
	elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
		add_precompiled_header(${the_target} ${pch_header})
	endif()
endif()

# For dynamic link numbering convenions
set_target_properties(${the_target} PROPERTIES
	VERSION ${CUDACNN_VERSION}
	SOVERSION ${CUDACNN_SOVERSION}
	OUTPUT_NAME "${the_target}${CUDACNN_DLLVERSION}"
	)
# Additional target properties
set_target_properties(${the_target} PROPERTIES
	DEBUG_POSTFIX "${CUDACNN_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
	INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
	)

# Add the required libraries for linking:
target_link_libraries(${the_target} ${CUDACNN_LINKER_LIBS})
if (HAVE_CUDA)
	target_link_libraries(${the_target} ${CUDA_LIBRARIES})
	CUDA_ADD_CUBLAS_TO_TARGET(${the_target})
endif()
if(HDF5_FOUND)
	target_link_libraries(${the_target} ${HDF5_LIBRARIES} )
endif()

if(MSVC)
	if(CMAKE_CROSSCOMPILING)
		set_target_properties(${the_target} PROPERTIES
			LINK_FLAGS "/NODEFAULTLIB:secchk"
			)
	endif()
	set_target_properties(${the_target} PROPERTIES
		LINK_FLAGS "/NODEFAULTLIB:libc"
		)
endif()
