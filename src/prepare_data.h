#pragma once
#ifndef PREPARE_DATA_H
#define PREPARE_DATA_H

#include "dataset_utils.h" // Für DatasetInfo & Encoding-Funktionen
#include <vector>
#include "Eigen/Eigen"

void prepare_training_data(
    const DatasetInfo& ds,
    float  val_split,
    std::vector<Eigen::VectorXd>& train_inputs,
    std::vector<Eigen::VectorXd>& train_labels_enc,
    std::vector<Eigen::VectorXd>& val_inputs,
    std::vector<Eigen::VectorXd>& val_labels_enc,
    std::vector<int>& train_labels_int,       // nur bei Klassifikation
    std::vector<int>& val_labels_int,
    std::vector<float>& train_labels_float,   // nur bei Regression
    std::vector<float>& val_labels_float
);
// Testdaten vorbereiten:
void prepare_test_data(
    const DatasetInfo& ds,
    std::vector<Eigen::VectorXd>& test_inputs,
    std::vector<Eigen::VectorXd>& test_labels_enc,
    std::vector<int>& test_labels_int,       // nur bei Klassifikation
    std::vector<float>& test_labels_float    // nur bei Regression
);

#endif // PREPARE_DATA_H