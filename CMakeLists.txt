cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(vscode-hpc CXX)

add_subdirectory(c++)

option(MCPI_ENABLE_CUDA "Enable CUDA" ON)
if (MCPI_ENABLE_CUDA)
  add_subdirectory(cuda)
endif ()