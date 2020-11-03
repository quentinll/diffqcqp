cmake_minimum_required(VERSION 3.0)

project(qcqplib)

find_package(Eigen3 3.3 REQUIRED)

set(SRCS
    Solver.cpp
    )
    
set(HEADERS
    Solver.hpp
    )

add_library(qcqplib ${SRCS} ${HEADERS})

target_link_libraries(qcqplib Eigen3::Eigen)
target_compile_options(qcqplib PRIVATE -fPIC)