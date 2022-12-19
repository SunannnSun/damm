# 2022, Sunan Sun <sunan@seas.upenn.edu>

cmake_minimum_required(VERSION 3.10)

# project(openMP_tutorial)
project(main)

set (CMAKE_CXX_STANDARD 14)

find_package(Boost REQUIRED COMPONENTS program_options random)
find_package(OpenMP REQUIRED)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)


include_directories(
    ../include
    ${PROJECT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR} 
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)



add_executable("${PROJECT_NAME}" main.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries("${PROJECT_NAME}" PUBLIC 
    OpenMP::OpenMP_CXX
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()


