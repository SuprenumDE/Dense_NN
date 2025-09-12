
// Noch in Arbeit!

#ifndef DIAGNOSTICS_UTILS_H
#define DIAGNOSTICS_UTILS_H

#include <vector>
#include "Eigen/Eigen"
#include <iostream>
#include <string>
#include <numeric>

// Zeigt die Norm des Gradienten für jede Schicht
void show_gradient_norms(const std::vector<Eigen::VectorXd>& deltas) {
    for (size_t l = 0; l < deltas.size(); ++l) {
        std::cout << "[Diagnose] Layer " << l << " - Gradienten-Norm: " << deltas[l].norm() << "\n";
    }
}

// Zeigt die Gewichtsnorm für jede Schicht
void show_weight_norms(const std::vector<Eigen::MatrixXd>& weights) {
    for (size_t l = 0; l < weights.size(); ++l) {
        std::cout << "[Diagnose] Layer " << l << " - Gewichtsnorm: " << weights[l].norm() << "\n";
    }
}

// Zeigt den Mittelwert der Biases für jede Schicht
void show_bias_means(const std::vector<Eigen::VectorXd>& biases) {
    for (size_t l = 0; l < biases.size(); ++l) {
        double mean = biases[l].mean();
        std::cout << "[Diagnose] Layer " << l << " - Bias-Mittelwert: " << mean << "\n";
    }
}

#endif // DIAGNOSTICS_UTILS_H
