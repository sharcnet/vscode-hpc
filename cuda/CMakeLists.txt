enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(mcpi-cuda mcpi-cuda.cu)
target_compile_options(mcpi-cuda PRIVATE
  "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_link_libraries(mcpi-cuda PUBLIC CUDA::curand)
