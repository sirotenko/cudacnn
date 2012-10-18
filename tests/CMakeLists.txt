project(tests)

set(the_target "tests")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/
	${CMAKE_SOURCE_DIR}/include
	${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
file(GLOB lib_srcs "*.cpp")
file(GLOB lib_int_hdrs "*.h*")

source_group("Src" FILES ${lib_srcs})
source_group("Include" FILES ${lib_int_hdrs})

if(NOT HAVE_BOOST)
	if(CMAKE_COMPILER_IS_GNUCXX)
	    #Enable C++11
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	endif() 
endif()


add_executable(${the_target} ${lib_srcs})
include_directories(${GTEST_INCLUDE_DIR})


if (HAVE_CUDA)		
	include_directories(${CUDA_INCLUDE_DIRS})	
endif()

if(PCHSupport_FOUND)
	set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/precomp.h)
	if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
		if(${CMAKE_GENERATOR} MATCHES "Visual*")
			set(${the_target}_pch "precomp.cpp")
		endif()            
		add_native_precompiled_header(${the_target} ${pch_header})
	elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
		add_precompiled_header(${the_target} ${pch_header})
	endif()
endif()

 		
# Add the required libraries for linking:
target_link_libraries(${the_target} cudacnn ${GTEST_LIBRARIES} ${CUDACNN_LINKER_LIBS} gtest gtest_main)
add_dependencies(${the_target} gtest)
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

