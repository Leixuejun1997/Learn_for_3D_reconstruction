cmake_minimum_required(VERSION 3.0.0)

project(TASK1-5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_BUILD_TYPE Debug)

include_directories(include)

add_executable(task1-5 task1-5_test_pose_from_fundamental.cpp)