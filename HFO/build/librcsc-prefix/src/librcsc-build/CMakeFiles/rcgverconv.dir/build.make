# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.19.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.19.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build

# Include any dependencies generated for this target.
include CMakeFiles/rcgverconv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rcgverconv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rcgverconv.dir/flags.make

CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o: CMakeFiles/rcgverconv.dir/flags.make
CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o: /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc/src/rcgverconv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o -c /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc/src/rcgverconv.cpp

CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc/src/rcgverconv.cpp > CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.i

CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc/src/rcgverconv.cpp -o CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.s

# Object files for target rcgverconv
rcgverconv_OBJECTS = \
"CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o"

# External object files for target rcgverconv
rcgverconv_EXTERNAL_OBJECTS =

bin/rcgverconv: CMakeFiles/rcgverconv.dir/src/rcgverconv.cpp.o
bin/rcgverconv: CMakeFiles/rcgverconv.dir/build.make
bin/rcgverconv: lib/librcsc_gz.a
bin/rcgverconv: lib/librcsc_rcg.a
bin/rcgverconv: CMakeFiles/rcgverconv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/rcgverconv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rcgverconv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rcgverconv.dir/build: bin/rcgverconv

.PHONY : CMakeFiles/rcgverconv.dir/build

CMakeFiles/rcgverconv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rcgverconv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rcgverconv.dir/clean

CMakeFiles/rcgverconv.dir/depend:
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build/CMakeFiles/rcgverconv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rcgverconv.dir/depend

