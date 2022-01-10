if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif ()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(mcpi_cuda mcpi_cuda.cu)
target_compile_options(mcpi_cuda PRIVATE
  "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_link_libraries(mcpi_cuda PUBLIC CUDA::curand)
