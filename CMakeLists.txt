cmake_minimum_required(VERSION 3.20)
project(CFD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall")

find_package(CUDA 10.0 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cfd main.cu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cfd PUBLIC OpenMP::OpenMP_CXX)
endif()

# Pass OpenMP flags to NVCC
target_compile_options(cfd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -fopenmp
                       >)