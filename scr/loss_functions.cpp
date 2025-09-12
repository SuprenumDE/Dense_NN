// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// NN Verlustfunktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.2, 27.07.2025
// --------------------------------------

#include "loss_functions.h"
#include <cmath>

double CrossEntropyLoss::compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const {
    // target ist ein One-Hot-Vektor
    double loss = 0.0;
    for (int i = 0; i < output.size(); ++i) {
        if (target[i] == 1.0) {
            loss = -std::log(std::max(output[i], 1e-12));
            break;
        }
    }
    return loss;
}


double MSELoss::compute(const Eigen::VectorXd& output, const Eigen::VectorXd& target) const {
    return (output - target).squaredNorm() / output.size();
}

// Funktion zum Erstellen der Loss-Funktion basierend auf dem LossType:
std::unique_ptr<LossFunction> createLossFunction(LossType type) {
    switch (type) {
    case LossType::CROSS_ENTROPY:
        return std::make_unique<CrossEntropyLoss>();
    case LossType::MSE:
        return std::make_unique<MSELoss>();
    case LossType::MAE:
        return std::make_unique<MAELoss>();
    default:
        throw std::invalid_argument("Unbekannter LossType");
    }
}

// Funktion zum Erstellen des Zielvektors für die Loss-Funktion:
Eigen::VectorXd LossFunction::getTarget(int label, size_t num_classes) const {
    Eigen::VectorXd target = Eigen::VectorXd::Zero(num_classes);
    if (label >= 0 && static_cast<size_t>(label) < num_classes) {
        target(label) = 1.0;
    }
    return target;
}

