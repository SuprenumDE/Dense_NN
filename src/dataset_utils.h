#pragma once
#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include <algorithm>   // für std::shuffle
#include <numeric>     // für std::iota
#include <random>
#include <vector>
#include <set>
#include <variant>
#include <string>
#include <iostream>
#include "Eigen/Eigen"

using LabelVariant = std::variant<int, float>;

struct DatasetInfo {
    int input_dim = -1;
    int output_dim = -1;
    bool is_classification = true;
    std::set<int> unique_labels;     // Nur für Klassifikation relevant
    std::size_t num_classes = 0;     // Anzahl der Klassen, nur für Klassifikation relevant

    std::vector<Eigen::VectorXd> inputs;
    std::vector<LabelVariant> labels;

    int get_label_as_int(std::size_t i) const {
        return std::get<int>(labels[i]);
    }

    float get_label_as_float(std::size_t i) const {
        return std::holds_alternative<int>(labels[i])
            ? static_cast<float>(std::get<int>(labels[i]))
            : std::get<float>(labels[i]);
    }

    Eigen::VectorXd get_encoded_label(std::size_t i) const {
        if (is_classification) {
            int label = std::get<int>(labels[i]);
            Eigen::VectorXd vec = Eigen::VectorXd::Zero(output_dim);
            vec(label) = 1.0;
            return vec;
        }
        else {
            return Eigen::VectorXd::Constant(1, get_label_as_float(i));
        }
    }
};

// Hilfsfunktionen

Eigen::MatrixXd vector2eigen_matrix(const std::vector<std::vector<double>>& data);

std::pair<float, Eigen::VectorXd> read_csv_row(const std::string& line);

DatasetInfo load_dataset_info(const std::string& filename, int max_samples = -1);

inline std::vector<int> get_labels_as_int_vector(const DatasetInfo& dataset) {
    std::vector<int> result;
    for (const auto& label : dataset.labels)
        result.push_back(std::get<int>(label));
    return result;
}

inline std::vector<float> get_labels_as_float_vector(const DatasetInfo& dataset) {
    std::vector<float> result;
    for (const auto& label : dataset.labels) {
        result.push_back(std::holds_alternative<int>(label)
            ? static_cast<float>(std::get<int>(label))
            : std::get<float>(label));
    }
    return result;
}

// Template-Funktion für generisches Splitten, es geht davon aus, dass die Labels entweder int oder float sind.
template <typename LabelType>
inline void split_data_generic(
    const std::vector<Eigen::VectorXd>& inputs,
    const std::vector<LabelType>& labels,
    float val_split,
    std::vector<Eigen::VectorXd>& train_inputs,
    std::vector<LabelType>& train_labels,
    std::vector<Eigen::VectorXd>& val_inputs,
    std::vector<LabelType>& val_labels) {

    train_inputs.clear(); val_inputs.clear();
    train_labels.clear(); val_labels.clear();

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    size_t val_size = static_cast<size_t>(inputs.size() * val_split);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto idx = indices[i];
        if (i < val_size) {
            val_inputs.push_back(inputs[idx]);
            val_labels.push_back(labels[idx]);
        }
        else {
            train_inputs.push_back(inputs[idx]);
            train_labels.push_back(labels[idx]);
        }
    }
}


// Wrapper-Funktionen split_data_classification und split_data_regression
inline void split_data_classification(
    const std::vector<Eigen::VectorXd>& inputs,
    const std::vector<int>& labels,
    float val_split,
    std::vector<Eigen::VectorXd>& train_inputs,
    std::vector<int>& train_labels,
    std::vector<Eigen::VectorXd>& val_inputs,
    std::vector<int>& val_labels) {

    split_data_generic<int>(inputs, labels, val_split, train_inputs, train_labels, val_inputs, val_labels);
}

inline void split_data_regression(
    const std::vector<Eigen::VectorXd>& inputs,
    const std::vector<float>& labels,
    float val_split,
    std::vector<Eigen::VectorXd>& train_inputs,
    std::vector<float>& train_labels,
    std::vector<Eigen::VectorXd>& val_inputs,
    std::vector<float>& val_labels) {

    split_data_generic<float>(inputs, labels, val_split, train_inputs, train_labels, val_inputs, val_labels);
}



// Variante mit kodierten Labels, die für Klassifikation verwendet werden.
// Sie wird für Klassifikationsprobleme verwendet, bei denen die Labels als One-Hot-Vektoren kodiert sind.
// (Vergleichbar mit Faktoren in R)
inline void split_encoded_labels(
    const std::vector<Eigen::VectorXd>& inputs,
    const std::vector<Eigen::VectorXd>& labels,
    float val_split,
    std::vector<Eigen::VectorXd>& train_inputs,
    std::vector<Eigen::VectorXd>& train_labels,
    std::vector<Eigen::VectorXd>& val_inputs,
    std::vector<Eigen::VectorXd>& val_labels)
{
    train_inputs.clear(); val_inputs.clear();
    train_labels.clear(); val_labels.clear();

    size_t val_size = static_cast<size_t>(inputs.size() * val_split);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i < val_size) {
            val_inputs.push_back(inputs[i]);
            val_labels.push_back(labels[i]);
        }
        else {
            train_inputs.push_back(inputs[i]);
            train_labels.push_back(labels[i]);
        }
    }
}


#endif // DATASET_UTILS_H
