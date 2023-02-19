
if(NOT "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitinfo.txt" IS_NEWER_THAN "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/local/bin/git"  clone --no-checkout "https://github.com/mhauskn/soccerwindow2.git" "soccerwindow2"
    WORKING_DIRECTORY "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/mhauskn/soccerwindow2.git'")
endif()

execute_process(
  COMMAND "/usr/local/bin/git"  checkout master --
  WORKING_DIRECTORY "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/local/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitinfo.txt"
    "/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/yueguo/Downloads/HFO/build/soccerwindow2-prefix/src/soccerwindow2-stamp/soccerwindow2-gitclone-lastrun.txt'")
endif()

