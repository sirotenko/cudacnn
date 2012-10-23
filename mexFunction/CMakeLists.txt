project(cudacnnMex)

set(the_target "cudacnnMex")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_int_hdrs "src/*.h*")
file(GLOB lib_hdrs "include/*.h*")

source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Include" FILES ${lib_hdrs})

if(NOT HAVE_BOOST)
	if(CMAKE_COMPILER_IS_GNUCXX)
	    #Enable C++11
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	endif() 
	if(CMAKE_GENERATOR STREQUAL Xcode)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
	endif()
endif()


add_library(${the_target} SHARED ${lib_srcs} ${lib_hdrs})
include_directories(${MATLAB_INCLUDE_DIR})


if (HAVE_CUDA)		
	include_directories(${CUDA_INCLUDE_DIRS})	
endif()

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
#set_target_properties(${the_target} PROPERTIES
#	VERSION ${CUDACNN_VERSION}
#	SOVERSION ${CUDACNN_SOVERSION}
#	OUTPUT_NAME "${the_target}${CUDACNN_DLLVERSION}"
#	)

 	
# Set appropriate extension for mex-file
set_target_properties(${the_target} PROPERTIES
	 SUFFIX ".${MATLAB_MEXFILE_EXT}"
	 PREFIX ""
	 )
# Don't get CMake confused about extension
set_target_properties(${the_target} PROPERTIES
	 LINKER_LANGUAGE CXX)
	
# Additional target properties
set_target_properties(${the_target} PROPERTIES
	DEBUG_POSTFIX "${CUDACNN_DEBUG_POSTFIX}"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
	INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/bin"
	)

# Add the required libraries for linking:
target_link_libraries(${the_target} cudacnn ${MATLAB_LIBRARIES} ${CUDACNN_LINKER_LIBS})

if(MSVC)
	if(CMAKE_CROSSCOMPILING)
		set_target_properties(${the_target} PROPERTIES
			LINK_FLAGS "/NODEFAULTLIB:secchk"
			)
	endif()
	set_target_properties(${the_target} PROPERTIES
		LINK_FLAGS "/NODEFAULTLIB:libc /EXPORT:mexFunction "
		)
endif()

install(TARGETS ${the_target}
	RUNTIME DESTINATION bin COMPONENT main
	LIBRARY DESTINATION lib COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)

install(FILES ${lib_hdrs}
	DESTINATION include/${name}
	COMPONENT main)

#After build copy mex-file to m_files
# get name and location of binary
GET_TARGET_PROPERTY(cudacnnMex_BIN_NAME cudacnnMex LOCATION)
ADD_CUSTOM_COMMAND(TARGET cudacnnMex
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${cudacnnMex_BIN_NAME} "${cudacnn_SOURCE_DIR}/m_files/@cnn/" )

