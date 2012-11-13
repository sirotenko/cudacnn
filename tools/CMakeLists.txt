project(cudacnnTools)

set(hdf5tobin "hdf5tobin")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")

if(NOT HAVE_BOOST)
	if(CMAKE_COMPILER_IS_GNUCXX)
	    #Enable C++11
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	endif() 
	if(CMAKE_GENERATOR STREQUAL Xcode)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
	endif()
endif()


add_executable(${hdf5tobin} "hdf5tobin.cpp")

if (HAVE_CUDA)		
	include_directories(${CUDA_INCLUDE_DIRS})	
endif()
 		
# Additional target properties
set_target_properties(${hdf5tobin} PROPERTIES
	DEBUG_POSTFIX "${CUDACNN_DEBUG_POSTFIX}"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
	INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/bin"
	)

# Add the required libraries for linking:
target_link_libraries(${hdf5tobin} cudacnn ${CUDACNN_LINKER_LIBS})


install(TARGETS ${hdf5tobin}
	RUNTIME DESTINATION bin COMPONENT main
	LIBRARY DESTINATION lib COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)

