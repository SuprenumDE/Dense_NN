// Dense neural network based on “Eigen”
// 
// Node initialization methods
// 
// Developer: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.3, 25.09.2025
// --------------------------------------

#pragma once
#ifndef WEIGHTS_INI_H
#define WEIGHTS_INI_H

#include "Eigen/Eigen"
#include <string>

// Enum for selecting the initialization method:
enum class InitType {
    XAVIER,
    HE,
    LECUN,
    ORTHOGONAL,
    UNIFORM,
    NORMAL,
    ITERATIVE
};

// Main function for initialization:
Eigen::MatrixXd initialize_weights(int output_size, int input_size, InitType method);

// Optional: Convert enum to string:
std::string to_string(InitType init);


#endif // WEIGHTS_IO_H 