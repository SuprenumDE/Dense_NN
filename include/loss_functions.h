#pragma once
#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

// NN Loss Functions
// 
// Dense neural network based on “Eigen”
// 
// Developer: Guenter Faes, spv@faes.de
// Version 0.0.3, 22.12.2025
// --------------------------------------



#include "Eigen/Eigen"
#include <memory>

enum class LossType {
    CROSS_ENTROPY,
    MSE,
    MAE
};

class LossFunction {
public:
    virtual double compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const = 0;
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const = 0;
    virtual ~LossFunction() = default;

    Eigen::VectorXd getTarget(int label, size_t num_classes) const;
};

class CrossEntropyLoss : public LossFunction {
public:
    double compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override;

    // Expected: target is a one-hot vector
    Eigen::VectorXd gradient(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override {
        return output - target;
    }
};


class MSELoss : public LossFunction {
public:
    double compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override;

	// Calculates the gradient of the MSE loss function:
    Eigen::VectorXd gradient(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override {
        return 2.0 * (output - target) / output.size();
    }

};

class MAELoss : public LossFunction {
public:
    double compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override {
        return (output - target).cwiseAbs().sum() / output.size();
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override {
        Eigen::VectorXd grad = output - target;
        for (int i = 0; i < grad.size(); ++i) {
            grad[i] = (grad[i] >= 0) ? 1.0 : -1.0;
        }
        return grad / output.size();
    }
};

// Convert enum to string and vice versa:
inline std::string to_string(LossType lt) {
    switch (lt) {
    case LossType::CROSS_ENTROPY: return "CROSS_ENTROPY";
    case LossType::MSE: return "MSE";
    case LossType::MAE: return "MAE";
    default: return "UNKNOWN";
    }
}

inline LossType loss_from_string(const std::string& s) {
    if (s == "CROSS_ENTROPY") return LossType::CROSS_ENTROPY;
    if (s == "MSE") return LossType::MSE;
    if (s == "MAE") return LossType::MAE;
    throw std::invalid_argument("Unknown LossType: " + s);
}


// Factory function for creating the loss function based on the LossType:
std::unique_ptr<LossFunction> createLossFunction(LossType type);

#endif // LOSS_FUNCTIONS_H
 