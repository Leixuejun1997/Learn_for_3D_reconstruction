# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lei/git_for_homework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lei/git_for_homework/build

# Include any dependencies generated for this target.
include task2/CMakeFiles/task2-1.dir/depend.make

# Include the progress variables for this target.
include task2/CMakeFiles/task2-1.dir/progress.make

# Include the compile flags for this target's objects.
include task2/CMakeFiles/task2-1.dir/flags.make

task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o: task2/CMakeFiles/task2-1.dir/flags.make
task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o: ../task2/task2-1_test_triangle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lei/git_for_homework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o"
	cd /home/lei/git_for_homework/build/task2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o -c /home/lei/git_for_homework/task2/task2-1_test_triangle.cpp

task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.i"
	cd /home/lei/git_for_homework/build/task2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lei/git_for_homework/task2/task2-1_test_triangle.cpp > CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.i

task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.s"
	cd /home/lei/git_for_homework/build/task2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lei/git_for_homework/task2/task2-1_test_triangle.cpp -o CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.s

# Object files for target task2-1
task2__1_OBJECTS = \
"CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o"

# External object files for target task2-1
task2__1_EXTERNAL_OBJECTS =

task2/task2-1: task2/CMakeFiles/task2-1.dir/task2-1_test_triangle.cpp.o
task2/task2-1: task2/CMakeFiles/task2-1.dir/build.make
task2/task2-1: task2/CMakeFiles/task2-1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lei/git_for_homework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable task2-1"
	cd /home/lei/git_for_homework/build/task2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task2-1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
task2/CMakeFiles/task2-1.dir/build: task2/task2-1

.PHONY : task2/CMakeFiles/task2-1.dir/build

task2/CMakeFiles/task2-1.dir/clean:
	cd /home/lei/git_for_homework/build/task2 && $(CMAKE_COMMAND) -P CMakeFiles/task2-1.dir/cmake_clean.cmake
.PHONY : task2/CMakeFiles/task2-1.dir/clean

task2/CMakeFiles/task2-1.dir/depend:
	cd /home/lei/git_for_homework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lei/git_for_homework /home/lei/git_for_homework/task2 /home/lei/git_for_homework/build /home/lei/git_for_homework/build/task2 /home/lei/git_for_homework/build/task2/CMakeFiles/task2-1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : task2/CMakeFiles/task2-1.dir/depend
