cmake_minimum_required(VERSION 3.20)
project(monte_carlo)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

# Apple Clang не постачає OpenMP — використовуємо libomp з Homebrew
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES   "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

add_executable(mc_serial serial.cpp)

add_executable(mc_mpi mpi_parallel.cpp)
target_link_libraries(mc_mpi MPI::MPI_CXX)

add_executable(mc_openmp openmp_parallel.cpp)
target_link_libraries(mc_openmp OpenMP::OpenMP_CXX)

add_executable(mc_launch_mpi launch/MPI.cpp)
target_link_libraries(mc_launch_mpi MPI::MPI_CXX)

add_executable(mc_serial_c monte-carlo/algorithm/monte_carlo.c)
target_link_libraries(mc_serial_c m)

add_executable(mc_mpi_c monte-carlo/cmd/mpi.c)
target_link_libraries(mc_mpi_c MPI::MPI_C m)

add_executable(mc_omp_c monte-carlo/cmd/open_mp.c)
target_link_libraries(mc_omp_c OpenMP::OpenMP_C m)
