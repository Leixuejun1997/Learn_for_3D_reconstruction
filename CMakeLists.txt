cmake_minimum_required(VERSION 3.0)

project(git_for_homework)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(task2)

add_subdirectory(include)

#add_subdirectory(task1)

add_subdirectory(task3)