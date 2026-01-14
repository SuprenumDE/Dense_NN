#pragma once
#ifndef METRICS_UTILS_H
#define METRICS_UTILS_H

#include <string>
#include <vector>
#include "config.h"                      // Config
#include "dataset_utils.h"               // DatasetInfo
#include "activation_functions.h"        // Activation::get()
#include "Eigen/Eigen"

// Trim-Funktion
std::string trim(const std::string& s);

// Argmax for classification
int argmax(const Eigen::VectorXd& vec);

// Dataset-Statistik ausgeben
void report_dataset_stats(const DatasetInfo& dataset, float val_split);

// Vorhersage durch das Netz
Eigen::VectorXd predict(
    const Eigen::VectorXd& input,
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const std::vector<std::string>& activations
);

// CSV-Speicherung der Vorhersagen
void save_predictions_csv(
    const std::string& filename,
    const std::vector<double>& true_values,
    const std::vector<double>& predicted_values
);

#endif // METRICS_UTILS_H