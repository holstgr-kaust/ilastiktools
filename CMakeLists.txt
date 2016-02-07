#----------------------------------------------------
# default build-type (release)
# (the next lines must be called bevore project(ilastiktools))
#----------------------------------------------------
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

#----------------------------------------------------
# ilastiktools
#----------------------------------------------------
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(ILASTIKTOOLS_ADD_NUMPY_MODULE)

project(ilastiktools)
set (ILASTIKTOOLS_VERSION_MAJOR 1)
set (ILASTIKTOOLS_VERSION_MINOR 0)
set (ILASTIKTOOLS_VERSION_PATCH 0)

#--------------------------------------------------------
# enable_testing
#--------------------------------------------------------
enable_testing()

#--------------------------------------------------------
# global headers
#--------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
include_directories(include)

# If libboost_python.so and libboost_python-mt.so exist on the same machine,
#  we can get into trouble if we don't link against the same version that vigra used.
#  (vigra uses the version without -mt)
# set(Boost_USE_MULTITHREADED OFF)

FIND_PACKAGE(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

FIND_PACKAGE(Threads REQUIRED)  # includes pthread
include(FindILASTIKTOOLS_DEPENDENCIES) # search for python, numpy, boost_python
include_directories(${ILASTIKTOOLS_INCLUDE_DIRS})

#--------------------------------------------------------
# OpenMP
#--------------------------------------------------------
option (WITH_OPENMP "use openmp" ON)
if(WITH_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  message(STATUS "build with OpenMP")
  add_definitions(-DWITH_OPENMP)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
   message(STATUS "build without openMP -> multithreaded options disabled")
endif(WITH_OPENMP)

#--------------------------------------------------------
# source directory
#--------------------------------------------------------
add_subdirectory(src)
