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

# Utility rule file for librcsc.

# Include the progress variables for this target.
include CMakeFiles/librcsc.dir/progress.make

CMakeFiles/librcsc: CMakeFiles/librcsc-complete


CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-install
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-mkdir
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-download
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-update
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-patch
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-configure
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-build
CMakeFiles/librcsc-complete: librcsc-prefix/src/librcsc-stamp/librcsc-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'librcsc'"
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/CMakeFiles
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/CMakeFiles/librcsc-complete
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-done

librcsc-prefix/src/librcsc-stamp/librcsc-install: librcsc-prefix/src/librcsc-stamp/librcsc-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'librcsc'"
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E echo_append
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-install

librcsc-prefix/src/librcsc-stamp/librcsc-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'librcsc'"
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/tmp
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E make_directory /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-mkdir

librcsc-prefix/src/librcsc-stamp/librcsc-download: librcsc-prefix/src/librcsc-stamp/librcsc-gitinfo.txt
librcsc-prefix/src/librcsc-stamp/librcsc-download: librcsc-prefix/src/librcsc-stamp/librcsc-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'librcsc'"
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src && /usr/local/Cellar/cmake/3.19.3/bin/cmake -P /Users/yueguo/Downloads/HFO/build/librcsc-prefix/tmp/librcsc-gitclone.cmake
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-download

librcsc-prefix/src/librcsc-stamp/librcsc-update: librcsc-prefix/src/librcsc-stamp/librcsc-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No update step for 'librcsc'"
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E echo_append
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-update

librcsc-prefix/src/librcsc-stamp/librcsc-patch: librcsc-prefix/src/librcsc-stamp/librcsc-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'librcsc'"
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E echo_append
	/usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-patch

librcsc-prefix/src/librcsc-stamp/librcsc-configure: librcsc-prefix/tmp/librcsc-cfgcmd.txt
librcsc-prefix/src/librcsc-stamp/librcsc-configure: librcsc-prefix/src/librcsc-stamp/librcsc-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Performing configure step for 'librcsc'"
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && /usr/local/Cellar/cmake/3.19.3/bin/cmake -DCMAKE_BUILD_TYPE=RelwithDebInfo "-GUnix Makefiles" /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-configure

librcsc-prefix/src/librcsc-stamp/librcsc-build: librcsc-prefix/src/librcsc-stamp/librcsc-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/yueguo/Downloads/HFO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Performing build step for 'librcsc'"
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && $(MAKE)
	cd /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-build && /usr/local/Cellar/cmake/3.19.3/bin/cmake -E touch /Users/yueguo/Downloads/HFO/build/librcsc-prefix/src/librcsc-stamp/librcsc-build

librcsc: CMakeFiles/librcsc
librcsc: CMakeFiles/librcsc-complete
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-build
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-configure
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-download
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-install
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-mkdir
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-patch
librcsc: librcsc-prefix/src/librcsc-stamp/librcsc-update
librcsc: CMakeFiles/librcsc.dir/build.make

.PHONY : librcsc

# Rule to build all files generated by this target.
CMakeFiles/librcsc.dir/build: librcsc

.PHONY : CMakeFiles/librcsc.dir/build

CMakeFiles/librcsc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/librcsc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/librcsc.dir/clean

CMakeFiles/librcsc.dir/depend:
	cd /Users/yueguo/Downloads/HFO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build /Users/yueguo/Downloads/HFO/build/CMakeFiles/librcsc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/librcsc.dir/depend

