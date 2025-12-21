#pragma once
#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

// NN Verlustfunktionen
// 
// Neuronales Netz für das MNIST-Beispiel
// auf Basis von "Eigen"
// 
// Entwickler: Guenter Faes, spv@faes.de
// Version 0.0.2, 27.07.2025
// --------------------------------------



#include "Eigen/Eigen"

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

    // Erwartet: target ist ein One-Hot-Vektor
    Eigen::VectorXd gradient(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override {
        return output - target;
    }
};


class MSELoss : public LossFunction {
public:
    double compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const override;

	// Berechnet den Gradient der MSE-Verlustfunktion
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

// Enum in String konvertieren und umgekehrt:
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


// Factory-Funktion zum Erstellen der Loss-Funktion basierend auf dem LossType
std::unique_ptr<LossFunction> createLossFunction(LossType type);

#endif // LOSS_FUNCTIONS_H
 