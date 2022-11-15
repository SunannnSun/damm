# 2022, Sunan Sun <sunan@seas.upenn.edu>

cmake_minimum_required(VERSION 3.10)

project(dpmm)

find_package(Boost REQUIRED COMPONENTS program_options random
)
# find_package (Eigen3 3.3 REQUIRED NO_MODULE
# )

message("My test's working directory:  ${Boost_INCLUDE_DIRS}")


include_directories(
    ../include
    ${PROJECT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}/eigen3.4
    ${EIGEN3_INCLUDE_DIR} 
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)


add_library(distribution_module
  # ${PROJECT_SOURCE_DIR}/niw.cpp
  ${PROJECT_SOURCE_DIR}/niw.cpp
  # ${PROJECT_SOURCE_DIR}/distribution.cpp
)


add_executable("${PROJECT_NAME}" dpmm.cpp
)


target_link_libraries("${PROJECT_NAME}"
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    distribution_module
)



# add_executable(main main.cpp
# )

# target_link_libraries(main
#     ${Boost_PROGRAM_OPTIONS_LIBRARY}
#     Eigen3::Eigen
# )