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
CMAKE_SOURCE_DIR = /Users/yueguo/Downloads/HFO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yueguo/Downloads/HFO/build

# Include any dependencies generated for this target.
include CMakeFiles/hfo_example_agent.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hfo_example_agent.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hfo_example_agent.dir/flags.make

CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o: CMakeFiles/hfo_example_agent.dir/flags.make
CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o: ../example/hfo_example_agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o -c /Users/yueguo/Downloads/HFO/example/hfo_example_agent.cpp

CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/example/hfo_example_agent.cpp > CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.i

CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/example/hfo_example_agent.cpp -o CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.s

# Object files for target hfo_example_agent
hfo_example_agent_OBJECTS = \
"CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o"

# External object files for target hfo_example_agent
hfo_example_agent_EXTERNAL_OBJECTS =

example/hfo_example_agent: CMakeFiles/hfo_example_agent.dir/example/hfo_example_agent.cpp.o
example/hfo_example_agent: CMakeFiles/hfo_example_agent.dir/build.make
example/hfo_example_agent: ../lib/libhfo.so
example/hfo_example_agent: libplayer_chain_action.a
example/hfo_example_agent: CMakeFiles/hfo_example_agent.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example/hfo_example_agent"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hfo_example_agent.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hfo_example_agent.dir/build: example/hfo_example_agent

.PHONY : CMakeFiles/hfo_example_agent.dir/build

CMakeFiles/hfo_example_agent.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hfo_example_agent.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hfo_example_agent.dir/clean

CMakeFiles/hfo_example_agent.dir/depend:
	cd /Users/yueguo/Downloads/HFO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build/CMakeFiles/hfo_example_agent.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hfo_example_agent.dir/depend

