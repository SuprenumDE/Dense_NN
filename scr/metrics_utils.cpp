// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// NN-Hilfsfuktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.1, 14.08.2025
// --------------------------------------

#include "metrics_utils.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>

Config config; // Globale Konfiguration, um auf die Modellart zuzugreifen

// Argmax für Klassifikation
int argmax(const Eigen::VectorXd& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}

// Trim-Funktion
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos || end == std::string::npos)
        ? ""
        : s.substr(start, end - start + 1);
}

// Dataset-Statistik ausgeben
void report_dataset_stats(const DatasetInfo& dataset, float val_split) {
    size_t total_samples = dataset.inputs.size();
    size_t val_samples = static_cast<size_t>(total_samples * val_split);
    size_t train_samples = total_samples - val_samples;

    std::cout << "\nDataset statistics:\n";
    std::cout << "  Total samples:             " << total_samples << "\n";
    std::cout << "  Training samples:          " << train_samples << "\n";
    std::cout << "  Validation samples:        " << val_samples << "\n";
    std::cout << "  Input dimension:           " << dataset.input_dim << "\n";
    std::cout << "  Classification:            " << (dataset.is_classification ? "Yes" : "No") << "\n";
    std::cout << "  Output dimension:          " << dataset.output_dim << "\n\n";
}

// Vorhersage durch das Netz
Eigen::VectorXd predict(
    const Eigen::VectorXd& input,
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const std::vector<std::string>& activations
) {
    // Debugging:
    assert(weights.size() == biases.size());
    assert(weights.size() == activations.size());

    Eigen::VectorXd a = input;
    for (size_t l = 0; l < weights.size(); ++l) {

        // Debugging:
        assert(weights[l].cols() == a.size());
        assert(weights[l].rows() == biases[l].size());

        Eigen::VectorXd z = weights[l] * a + biases[l];
        auto act = Activation::get(activations[l]);
        a = act->activate(z);
    }
    return a;
}

// CSV-Speicherung der Vorhersagen
void save_predictions_csv(
    const std::string& filename,
    const std::vector<double>& true_values,
    const std::vector<double>& predicted_values
) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening the file: " << filename << std::endl;
        return;
    }

    std::string header_true, header_pred;
    if (config.model_type == ModelType::CLASSIFICATION) {
        header_true = "true_label";
        header_pred = "predicted_label";
    }
    else {
        header_true = "true_value";
        header_pred = "predicted_value";
    }

    out << header_true << "," << header_pred << "\n";
    for (size_t i = 0; i < true_values.size(); ++i) {
        out << true_values[i] << "," << predicted_values[i] << "\n";
    }

    out.close();
}
