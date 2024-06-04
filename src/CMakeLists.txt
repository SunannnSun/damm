cmake_minimum_required(VERSION 3.16)

project(Damm VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    message("Apple (macOS) detected")
    # change default compiler to llvm
    set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
    set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)
    set(CMAKE_PREFIX_PATH /opt/homebrew/opt/libomp)
endif()

if(LINUX) 
    message("Ubuntu detected")
    set(CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17") # for gcc in linux
endif()



# Display the compiler being used
message("Using C++ compiler: ${CMAKE_CXX_COMPILER}")
message("Project Root Directory: ${CMAKE_SOURCE_DIR}")


find_package(Eigen3 3.4 REQUIRED)
message("Eigen3 Include Directory: ${EIGEN3_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS program_options)
message("Boost Include Directory: ${Boost_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
message("OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")

find_package(OpenMP REQUIRED)
message(STATUS "OpenMP Include Directory: ${OpenMP_CXX_INCLUDE_DIRS}")



add_executable(main main.cpp niw.cpp niwDamm.cpp gauss.cpp gaussDamm.cpp dpmm.cpp damm.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

target_compile_options(main PRIVATE -fopenmp)

set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/../include)
# target_include_directories(main PRIVATE /usr/include/eigen-3.4.0)

target_include_directories(main PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})


target_link_libraries(main PRIVATE Boost::program_options)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)






