cmake_minimum_required(VERSION 3.11)

project(DALang VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("-- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Build Debug
if("${DEBUG}" STREQUAL "on")
set(CMAKE_BUILD_TYPE "Debug")
else()
set(CMAKE_BUILD_TYPE "Release")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
# Set PIC option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # add_compile_options(-fPIC)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/dair)
include_directories(${PROJECT_SOURCE_DIR}/dart)

# Add ir, and runtime directories
add_subdirectory(dair)
add_subdirectory(dart)

set(DALANG_OBJ tensor_obj runtime_obj ops_obj pass_obj dl)

add_library(dalang INTERFACE)
target_link_libraries(dalang INTERFACE ${DALANG_OBJ})
