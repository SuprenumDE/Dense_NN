// // Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// NN Aktivierungsfunktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.2, 20.07.2025
// --------------------------------------

#include "activation_functions.h"
#include <stdexcept>
#include <cmath>

// ReLU
Eigen::VectorXd ReluActivation::activate(const Eigen::VectorXd& x) const {
    return x.array().max(0.0);
}
Eigen::VectorXd ReluActivation::derivative(const Eigen::VectorXd& x) const {
    return (x.array() > 0.0).cast<double>();
}

// Sigmoid
Eigen::VectorXd SigmoidActivation::activate(const Eigen::VectorXd& x) const {
    return 1.0 / (1.0 + (-x.array()).exp());
}
Eigen::VectorXd SigmoidActivation::derivative(const Eigen::VectorXd& x) const {
    Eigen::VectorXd s = activate(x);
    return s.array() * (1.0 - s.array());
}

// Tanh
Eigen::VectorXd TanhActivation::activate(const Eigen::VectorXd& x) const {
    return x.array().tanh();
}
Eigen::VectorXd TanhActivation::derivative(const Eigen::VectorXd& x) const {
    return 1.0 - x.array().tanh().square();
}

// Softmax
Eigen::VectorXd SoftmaxActivation::activate(const Eigen::VectorXd& x) const {
    Eigen::ArrayXd exps = (x.array() - x.maxCoeff()).exp();
    return exps / exps.sum();
}
Eigen::VectorXd SoftmaxActivation::derivative(const Eigen::VectorXd& x) const {
    // Vereinfachte Form, echte Ableitung oft zusammen mit Cross-Entropy behandelt
    return Eigen::VectorXd::Ones(x.size());
}

// Identity
Eigen::VectorXd IdentityActivation::activate(const Eigen::VectorXd& x) const {
    return x;
}
Eigen::VectorXd IdentityActivation::derivative(const Eigen::VectorXd& x) const {
    return Eigen::VectorXd::Ones(x.size());
}

// Factory
std::shared_ptr<ActivationFunction> Activation::get(const std::string& name) {
    if (name == "relu") return std::make_shared<ReluActivation>();
    if (name == "sigmoid") return std::make_shared<SigmoidActivation>();
    if (name == "tanh") return std::make_shared<TanhActivation>();
    if (name == "softmax") return std::make_shared<SoftmaxActivation>();
    if (name == "none") return std::make_shared<IdentityActivation>();
    throw std::invalid_argument("Unknown activation function: " + name);
}