#pragma once

// NN optimizer functions
// Dense neural network based on “Eigen”
//
// Developer: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// Lizenz: MIT (auch für alle selbst entwickelten h- und cpp-Dateien)
// Version 0.0.1, 11.11.2025
// --------------------------------------

#include "Eigen/Dense"
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ------------------- Optimizer types -------------------
enum class OptimizerType { SGD, RMSProp, Adam };

// ------------------- Hyperparameter -------------------
struct OptimizerParams {
    OptimizerType type = OptimizerType::SGD;
    double beta1 = 0.9;    // Adam
    double beta2 = 0.999;  // Adam
    double epsilon = 1e-8;   // Adam/RMSProp
    double rho = 0.9;    // RMSProp
};

// ------------------- States -------------------
struct OptimizerState {
    // RMSProp:
    std::vector<Eigen::MatrixXd> G_W;
    std::vector<Eigen::VectorXd> G_b;

    // Adam:
    std::vector<Eigen::MatrixXd> M_W, V_W;
    std::vector<Eigen::VectorXd> M_b, V_b;

    // Counter for Adam:
    long long t = 0;
};

// ------------------- Interfaces -------------------

// String -> OptimizerType
OptimizerType parse_optimizer(const std::string& s);

// Initialization of states appropriate to the architecture:
void init_optimizer_state(
    OptimizerState& state,
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const OptimizerParams& params);

// Parameter update (in layers) with external learning rate:
void apply_update(
    std::vector<Eigen::MatrixXd>& weights,
    std::vector<Eigen::VectorXd>& biases,
    const std::vector<Eigen::MatrixXd>& dW,
    const std::vector<Eigen::VectorXd>& db,
    OptimizerState& state,
    const OptimizerParams& params,
    double learning_rate);

// Convert enum to string:
inline std::string to_string(OptimizerType opt) {
    switch (opt) {
    case OptimizerType::SGD:     return "SGD";
    case OptimizerType::RMSProp: return "RMSProp";
    case OptimizerType::Adam:    return "Adam";
    default:                     return "Unknown";
    }
}

