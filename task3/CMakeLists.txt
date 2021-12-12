cmake_minimum_required(VERSION 3.0)


include_directories("../include")

set(INCREMENTAL_SFM_SOURCES
        task3-1_test_incremental_sfm.cpp defines.h functions.cc functions.h)
add_executable(task3-1 ${INCREMENTAL_SFM_SOURCES})
target_link_libraries(task3-1 sfm util core features)