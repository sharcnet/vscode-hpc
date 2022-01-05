# check/add compiler OpenMP support flags
find_package(OpenMP)

add_executable(mcpi_seq mcpi_seq.cpp)
target_compile_features(mcpi_seq PRIVATE cxx_std_11)
install(TARGETS mcpi_seq RUNTIME DESTINATION bin COMPONENT mcpi_seq)
