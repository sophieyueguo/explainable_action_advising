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
include CMakeFiles/sample_player.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sample_player.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sample_player.dir/flags.make

CMakeFiles/sample_player.dir/src/HFO.cpp.o: CMakeFiles/sample_player.dir/flags.make
CMakeFiles/sample_player.dir/src/HFO.cpp.o: ../src/HFO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sample_player.dir/src/HFO.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_player.dir/src/HFO.cpp.o -c /Users/yueguo/Downloads/HFO/src/HFO.cpp

CMakeFiles/sample_player.dir/src/HFO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_player.dir/src/HFO.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/src/HFO.cpp > CMakeFiles/sample_player.dir/src/HFO.cpp.i

CMakeFiles/sample_player.dir/src/HFO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_player.dir/src/HFO.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/src/HFO.cpp -o CMakeFiles/sample_player.dir/src/HFO.cpp.s

CMakeFiles/sample_player.dir/src/main_player.cpp.o: CMakeFiles/sample_player.dir/flags.make
CMakeFiles/sample_player.dir/src/main_player.cpp.o: ../src/main_player.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sample_player.dir/src/main_player.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_player.dir/src/main_player.cpp.o -c /Users/yueguo/Downloads/HFO/src/main_player.cpp

CMakeFiles/sample_player.dir/src/main_player.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_player.dir/src/main_player.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/src/main_player.cpp > CMakeFiles/sample_player.dir/src/main_player.cpp.i

CMakeFiles/sample_player.dir/src/main_player.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_player.dir/src/main_player.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/src/main_player.cpp -o CMakeFiles/sample_player.dir/src/main_player.cpp.s

CMakeFiles/sample_player.dir/src/sample_player.cpp.o: CMakeFiles/sample_player.dir/flags.make
CMakeFiles/sample_player.dir/src/sample_player.cpp.o: ../src/sample_player.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sample_player.dir/src/sample_player.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_player.dir/src/sample_player.cpp.o -c /Users/yueguo/Downloads/HFO/src/sample_player.cpp

CMakeFiles/sample_player.dir/src/sample_player.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_player.dir/src/sample_player.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/src/sample_player.cpp > CMakeFiles/sample_player.dir/src/sample_player.cpp.i

CMakeFiles/sample_player.dir/src/sample_player.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_player.dir/src/sample_player.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/src/sample_player.cpp -o CMakeFiles/sample_player.dir/src/sample_player.cpp.s

CMakeFiles/sample_player.dir/src/agent.cpp.o: CMakeFiles/sample_player.dir/flags.make
CMakeFiles/sample_player.dir/src/agent.cpp.o: ../src/agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sample_player.dir/src/agent.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_player.dir/src/agent.cpp.o -c /Users/yueguo/Downloads/HFO/src/agent.cpp

CMakeFiles/sample_player.dir/src/agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_player.dir/src/agent.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yueguo/Downloads/HFO/src/agent.cpp > CMakeFiles/sample_player.dir/src/agent.cpp.i

CMakeFiles/sample_player.dir/src/agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_player.dir/src/agent.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yueguo/Downloads/HFO/src/agent.cpp -o CMakeFiles/sample_player.dir/src/agent.cpp.s

# Object files for target sample_player
sample_player_OBJECTS = \
"CMakeFiles/sample_player.dir/src/HFO.cpp.o" \
"CMakeFiles/sample_player.dir/src/main_player.cpp.o" \
"CMakeFiles/sample_player.dir/src/sample_player.cpp.o" \
"CMakeFiles/sample_player.dir/src/agent.cpp.o"

# External object files for target sample_player
sample_player_EXTERNAL_OBJECTS =

teams/base/sample_player: CMakeFiles/sample_player.dir/src/HFO.cpp.o
teams/base/sample_player: CMakeFiles/sample_player.dir/src/main_player.cpp.o
teams/base/sample_player: CMakeFiles/sample_player.dir/src/sample_player.cpp.o
teams/base/sample_player: CMakeFiles/sample_player.dir/src/agent.cpp.o
teams/base/sample_player: CMakeFiles/sample_player.dir/build.make
teams/base/sample_player: libplayer_chain_action.a
teams/base/sample_player: CMakeFiles/sample_player.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable teams/base/sample_player"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_player.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sample_player.dir/build: teams/base/sample_player

.PHONY : CMakeFiles/sample_player.dir/build

CMakeFiles/sample_player.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sample_player.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sample_player.dir/clean

CMakeFiles/sample_player.dir/depend:
	cd /Users/yueguo/Downloads/HFO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build/CMakeFiles/sample_player.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sample_player.dir/depend

