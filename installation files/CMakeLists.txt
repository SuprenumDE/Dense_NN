# ----------------------------------------------------------
#            EigenNET Procet / Dense_NN
#
# Compiling Dense_NN on the Raspberry Pi5
# EigenNET-Project: https://github.com/SuprenumDE/Dense_NN
# Guenter Faes, eigennet@faes.de
# Date: 2025-12-21
# ----------------------------------------------------------

cmake_minimum_required(VERSION 3.25)

project(DenseNN_RPi5 VERSION 0.1.20 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

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


# --- Compiler options for ARM Cortex-A76 ---
# Arguments:
#    -03: Maximum optimization, e.g., vector optimization
#    -march=native: Code is optimized for this specific CPUCode is optimized for this specific CPU
#    -ffast-math: “Turbo mode” for numerical calculations: Use of faster but less accurate instructions,
#                 vector optimizations that would otherwise be prohibited
#    -pthread: Enable threading support, OpenMP (indirectly), Eigen‑Multithreading 
target_compile_options(Dense_NN PRIVATE -O3 -march=native -ffast-math -pthread)

target_link_libraries(Dense_NN PRIVATE Eigen3::Eigen)

# ============================================================
# CONFIG-PACKAGE (Export Targets + Config + Version)
# ============================================================

# 1. Install executable + export target
install(TARGETS Dense_NN
    EXPORT DenseNNTargets
    RUNTIME DESTINATION bin
)

# 2. Install exported targets file
install(EXPORT DenseNNTargets
    FILE DenseNNTargets.cmake
    NAMESPACE DenseNN::
    DESTINATION lib/cmake/DenseNN
)

# 3. Generate Config file from template
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/DenseNNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DenseNNConfig.cmake
    INSTALL_DESTINATION lib/cmake/DenseNN
) 

# 4. Generate Version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DenseNNConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# 5. Install Config + Version files
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/DenseNNConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DenseNNConfigVersion.cmake
        DESTINATION lib/cmake/DenseNN
)


# ============================================================
# INSTALL HEADERS
# ============================================================
install(DIRECTORY include/ DESTINATION include)





