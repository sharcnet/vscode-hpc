# check/add compiler OpenMP support flags
find_package(OpenMP)
find_package(MPI)

add_executable(mcpi_seq mcpi_seq.cpp)
target_compile_features(mcpi_seq PRIVATE cxx_std_11)
install(TARGETS mcpi_seq RUNTIME DESTINATION bin COMPONENT mcpi_seq)

if (OpenMP_FOUND)
  add_executable(mcpi_par mcpi_par.cpp)
  target_compile_features(mcpi_par PRIVATE cxx_std_11)
  target_link_libraries(mcpi_par PUBLIC OpenMP::OpenMP_CXX)
  install(TARGETS mcpi_par RUNTIME DESTINATION bin COMPONENT mcpi_par)
endif ()

if (MPI_FOUND)
  add_executable(mcpi_mpi mcpi_mpi.cpp)
  target_compile_features(mcpi_mpi PRIVATE cxx_std_11)
  target_link_libraries(mcpi_mpi PUBLIC MPI::MPI_CXX)
  install(TARGETS mcpi_mpi RUNTIME DESTINATION bin COMPONENT mcpi_mpi)
endif ()