cmake_minimum_required(VERSION 3.0)
project(TASK2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_BUILD_TYPE Debug)
include_directories(include)
add_executable(task2-1 task2-1_test_triangle.cpp)

