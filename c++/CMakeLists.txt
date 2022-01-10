# check/add compiler OpenMP support flags
find_package(OpenMP)
find_package(MPI)

add_executable(mcpi-seq mcpi-seq.cpp)
target_compile_features(mcpi-seq PRIVATE cxx_std_11)
install(TARGETS mcpi-seq RUNTIME DESTINATION bin COMPONENT mcpi-seq)

if (OpenMP_FOUND)
  add_executable(mcpi-par mcpi-par.cpp)
  target_compile_features(mcpi-par PRIVATE cxx_std_11)
  target_link_libraries(mcpi-par PUBLIC OpenMP::OpenMP_CXX)
  install(TARGETS mcpi-par RUNTIME DESTINATION bin COMPONENT mcpi-par)
endif ()

if (MPI_FOUND)
  add_executable(mcpi-mpi mcpi-mpi.cpp)
  target_compile_features(mcpi-mpi PRIVATE cxx_std_11)
  target_link_libraries(mcpi-mpi PUBLIC MPI::MPI_CXX)
  install(TARGETS mcpi-mpi RUNTIME DESTINATION bin COMPONENT mcpi-mpi)
endif ()
