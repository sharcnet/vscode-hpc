cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(vscode-hpc CXX)

add_subdirectory(c++)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()