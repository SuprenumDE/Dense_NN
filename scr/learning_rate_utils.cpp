// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Lernraten-Hilfsfunktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.1, 25.06.2025
// --------------------------------------

#include "learning_rate_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

double adjust_learning_rate(const std::string& lr_mode, double base_lr, int epoch) {
    std::string mode = lr_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

    if (mode == "none" || mode == "") {
        return base_lr;
    }
    else if (mode == "decay") {
        return std::max(base_lr * std::pow(0.95, epoch), 1e-4);
    }
    else if (mode == "step") {
        return ((epoch + 1) % 10 == 0) ? base_lr * 0.5 : base_lr;
    }
    else {
        std::cerr << "[Warning] Unknown lr_mode: \"" << lr_mode << "\" - fixed learning rate is used.\n";
        return base_lr;
    }
}
