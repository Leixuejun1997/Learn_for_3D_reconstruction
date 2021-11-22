cmake_minimum_required(VERSION 3.0)

project(TASK1-6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_BUILD_TYPE Debug)#设置编译类型为debug
#添加头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

#这里不先生成相应对的库文件，后面就找不到
add_subdirectory(include/core)
add_subdirectory(include/util)
add_subdirectory(include/features)
add_subdirectory(include/sfm)

add_executable(task1-7 task1-7_test_features.cpp visualizer.cc visualizer.h)
target_link_libraries(task1-7 sfm util core features)

add_executable(task1-6 task1-6_test_matching.cpp visualizer.cc visualizer.h) 

target_link_libraries(task1-6 sfm util core features)

