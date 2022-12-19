# 2022, Sunan Sun <sunan@seas.upenn.edu>

cmake_minimum_required(VERSION 3.10)

project(openMP_tutorial)
set (CMAKE_CXX_STANDARD 14)


add_executable("${PROJECT_NAME}" openMP_tutorial.cpp
)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries("${PROJECT_NAME}" PUBLIC OpenMP::OpenMP_CXX)
endif()
