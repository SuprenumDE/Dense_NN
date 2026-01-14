# ----------------------------------------------------------
#            EigenNET Procet / Dense_NN
#
# Compiling Dense_NN on the Raspberry Pi5
# EigenNET-Project: https://github.com/SuprenumDE/Dense_NN
# Guenter Faes, eigennet@faes.de
# Date: 2026-01-10
# ----------------------------------------------------------

cmake_minimum_required(VERSION 3.25)

project(DenseNN_RPi5 VERSION 0.1.30 LANGUAGES CXX)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCES
    src/train_nn.cpp
    src/activation_functions.cpp
    src/config.cpp
    src/config_utils.cpp
    src/dataset_utils.cpp
    src/learning_rate_utils.cpp
    src/logging_utils.cpp
    src/loss_functions.cpp
    src/metrics_utils.cpp
    src/minibatch_parallel.cpp
    src/optimizer.cpp
    src/prepare_data.cpp
    src/weights_ini.cpp
    src/weights_io.cpp
)

add_executable(Dense_NN ${SOURCES})

target_include_directories(Dense_NN PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(Dense_NN PRIVATE -O3 -march=native -ffast-math -pthread)

target_link_libraries(Dense_NN PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    gomp
)




