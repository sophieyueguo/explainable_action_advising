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
CMAKE_SOURCE_DIR = /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build

# Utility rule file for soccerwindow2_autogen.

# Include the progress variables for this target.
include CMakeFiles/soccerwindow2_autogen.dir/progress.make

CMakeFiles/soccerwindow2_autogen:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic MOC for target soccerwindow2"
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E cmake_autogen /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build/CMakeFiles/soccerwindow2_autogen.dir/AutogenInfo.json MinSizeRel

soccerwindow2_autogen: CMakeFiles/soccerwindow2_autogen
soccerwindow2_autogen: CMakeFiles/soccerwindow2_autogen.dir/build.make

.PHONY : soccerwindow2_autogen

# Rule to build all files generated by this target.
CMakeFiles/soccerwindow2_autogen.dir/build: soccerwindow2_autogen

.PHONY : CMakeFiles/soccerwindow2_autogen.dir/build

CMakeFiles/soccerwindow2_autogen.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/soccerwindow2_autogen.dir/cmake_clean.cmake
.PHONY : CMakeFiles/soccerwindow2_autogen.dir/clean

CMakeFiles/soccerwindow2_autogen.dir/depend:
	cd /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2 /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2 /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build /Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-build/CMakeFiles/soccerwindow2_autogen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/soccerwindow2_autogen.dir/depend

