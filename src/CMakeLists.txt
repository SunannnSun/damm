cmake_minimum_required(VERSION 3.12)

project(DAMM VERSION 1.0)

set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(ENV{CXX} "/opt/homebrew/opt/llvm/bin/clang++")
set(ENV{LDFLAGS} "-L/opt/homebrew/opt/llvm/lib")
set(ENV{CPPFLAGS} "-I/opt/homebrew/opt/llvm/include")

find_package(Eigen3 REQUIRED)
message("Eigen3 Include Directory: ${EIGEN3_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS program_options)
message("Boost Include Directory: ${Boost_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
message("OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")

# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#     message(STATUS "OpenMP Include Directory: ${OpenMP_CXX_INCLUDE_DIRS}")
# endif()



message("Project Root Directory: ${CMAKE_SOURCE_DIR}")

add_executable(main main.cpp niw.cpp niwDir.cpp normal.cpp normalDir.cpp dpmm.cpp dpmmDir.cpp spectral.cpp)
target_compile_options(main PRIVATE -fopenmp)

set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..)

include_directories(${CMAKE_SOURCE_DIR}/../include)
target_include_directories(main PRIVATE /opt/homebrew/Cellar/libomp/16.0.6/include)
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(main PRIVATE Boost::program_options)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_link_libraries(main PRIVATE /opt/homebrew/Cellar/libomp/16.0.6/lib/libomp.dylib)







