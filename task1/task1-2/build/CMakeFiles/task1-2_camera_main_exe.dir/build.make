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
CMAKE_SOURCE_DIR = /home/lei/learn_SFM/task1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lei/learn_SFM/task1/build

# Include any dependencies generated for this target.
include CMakeFiles/task1-2_camera_main_exe.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/task1-2_camera_main_exe.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task1-2_camera_main_exe.dir/flags.make

CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o: CMakeFiles/task1-2_camera_main_exe.dir/flags.make
CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o: ../task1-2_test_camera_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lei/learn_SFM/task1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o -c /home/lei/learn_SFM/task1/task1-2_test_camera_main.cpp

CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lei/learn_SFM/task1/task1-2_test_camera_main.cpp > CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.i

CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lei/learn_SFM/task1/task1-2_test_camera_main.cpp -o CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.s

CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o: CMakeFiles/task1-2_camera_main_exe.dir/flags.make
CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o: ../src/task1-2_camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lei/learn_SFM/task1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o -c /home/lei/learn_SFM/task1/src/task1-2_camera.cpp

CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lei/learn_SFM/task1/src/task1-2_camera.cpp > CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.i

CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lei/learn_SFM/task1/src/task1-2_camera.cpp -o CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.s

# Object files for target task1-2_camera_main_exe
task1__2_camera_main_exe_OBJECTS = \
"CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o" \
"CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o"

# External object files for target task1-2_camera_main_exe
task1__2_camera_main_exe_EXTERNAL_OBJECTS =

task1-2_camera_main_exe: CMakeFiles/task1-2_camera_main_exe.dir/task1-2_test_camera_main.cpp.o
task1-2_camera_main_exe: CMakeFiles/task1-2_camera_main_exe.dir/src/task1-2_camera.cpp.o
task1-2_camera_main_exe: CMakeFiles/task1-2_camera_main_exe.dir/build.make
task1-2_camera_main_exe: CMakeFiles/task1-2_camera_main_exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lei/learn_SFM/task1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable task1-2_camera_main_exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task1-2_camera_main_exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task1-2_camera_main_exe.dir/build: task1-2_camera_main_exe

.PHONY : CMakeFiles/task1-2_camera_main_exe.dir/build

CMakeFiles/task1-2_camera_main_exe.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task1-2_camera_main_exe.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task1-2_camera_main_exe.dir/clean

CMakeFiles/task1-2_camera_main_exe.dir/depend:
	cd /home/lei/learn_SFM/task1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lei/learn_SFM/task1 /home/lei/learn_SFM/task1 /home/lei/learn_SFM/task1/build /home/lei/learn_SFM/task1/build /home/lei/learn_SFM/task1/build/CMakeFiles/task1-2_camera_main_exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/task1-2_camera_main_exe.dir/depend
