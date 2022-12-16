cmake_minimum_required(VERSION 3.10)

project(numerical_optimization)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

add_executable(numerical_optimization main.cpp)
target_link_libraries(numerical_optimization ${CERES_LIBRARIES})