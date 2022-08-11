cmake_minimum_required(VERSION 3.20)
project(CFD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenMP)

add_executable(cfd main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cfd PUBLIC OpenMP::OpenMP_CXX)
endif()