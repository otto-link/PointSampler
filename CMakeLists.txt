cmake_minimum_required(VERSION 3.12)
project(point_sampler-root VERSION 0.0.0)

# --- global setup

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if(UNIX AND NOT APPLE)
  message(STATUS "Configuring for LINUX system...")
  add_definitions("-DPS_OS_LINUX")
else()
  message(STATUS "NOT configuring for LINUX system...")
endif()

# --- release type

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting default build type to Release")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Release type: ${CMAKE_BUILD_TYPE}")

add_compile_definitions($<$<CONFIG:Debug>:DEBUG_BUILD>
                        $<$<CONFIG:Release>:RELEASE_BUILD>)

# --- compiler options

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # MSVC compiler options
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} /W4 /Od /D M_PI=3.14159265358979323846")
else()
  # GNU compiler options

  # Common to all configs
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wall -Wextra -Wno-dangling-reference -pthread -Wno-deprecated-declarations"
  )

  # Add per-configuration flags
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DLOG_LEVEL=3")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DLOG_LEVEL=3")
endif()

# --- dependencies

add_subdirectory(PointSampler)
add_subdirectory(external)
add_subdirectory(tests)
