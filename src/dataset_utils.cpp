// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Datenhandling-Funktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.4, 22.08.2025
// --------------------------------------


#include "dataset_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <string>   
#include <vector>
#include <set>
#include <cmath>

using namespace std;

// Funktion zur Wandlung von Vektor zu Eigen-Matrix:
Eigen::MatrixXd vector2eigen_matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("Pass empty data structure.");
    }

    int rows = static_cast<int>(data.size());
    int cols = static_cast<int>(data[0].size());
    Eigen::MatrixXd mat(rows, cols);

    for (int i = 0; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::invalid_argument("Inconsistent line lengths in the vector.");
        }
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }

    return mat;
}



// Funktion zum Einlesen einer CSV-Zeile und Umwandlung in float-Werte
std::pair<float, Eigen::VectorXd> read_csv_row(const std::string& line) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> values;


    while (std::getline(ss, cell, ',')) {
        try {
            values.push_back(std::stof(cell));
        }
        catch (const std::invalid_argument&) {
            std::cerr << "Invalid value in CSV line: \"" << cell << "\"" << std::endl;
        }
        catch (const std::out_of_range&) {
            std::cerr << "Numerical value outside the representable range: \"" << cell << "\"" << std::endl;
        }
    }

    if (values.size() < 2) {
        std::cerr << "Line contains too few valid values. Skipped.\n";
        return { 0.0f, Eigen::VectorXd() };
    }

    float label = values.front();
    Eigen::VectorXd input(values.size() - 1);
    for (size_t i = 1; i < values.size(); ++i) {
        input[i - 1] = values[i];
    }

    return { label, input };
}

// Funktion zum Einlesen der Dataset-Informationen aus einer CSV-Datei:
DatasetInfo load_dataset_info(const std::string& filename, int max_samples) {

    if (!std::filesystem::exists(filename) || !std::filesystem::is_regular_file(filename)) {
        std::cerr << "File not found or not a regular file: "
            << filename << std::endl;
        throw std::runtime_error("Invalid dataset file");
    }

    ifstream file(filename);
    string line;
    int count = 0;

    std::vector<Eigen::VectorXd> inputs;
    std::vector<LabelVariant> labels;
    std::set<int> unique_labels;
    bool is_classification = true;
    int input_dim = -1;

    while (std::getline(file, line) && (max_samples < 0 || count < max_samples)) {
        auto [label, input] = read_csv_row(line);

        if (input.size() == 0) continue;

        if (input_dim == -1) input_dim = input.size();

        inputs.push_back(input);
        LabelVariant vlabel = is_classification
            ? LabelVariant(static_cast<int>(label))
            : LabelVariant(label);
        labels.push_back(vlabel);


        if (floor(label) == label)
            unique_labels.insert(static_cast<int>(label));
        else
            is_classification = false;

        ++count;
    }

    DatasetInfo info;

	info.input_dim = input_dim;                     // Setzen der Eingabedimension
	info.is_classification = is_classification;     // Setzen des Klassifikationsmodus
	info.unique_labels = unique_labels;             // Speichern der eindeutigen Labels


    if (is_classification) {
        info.num_classes = static_cast<int>(unique_labels.size());
        info.output_dim = info.num_classes;                        // Anzahl der Klassen
    }
    else {
        info.num_classes = 0;
        info.output_dim = 1;                                      // Skalar für Regression
    }

	// Daten in DatasetInfo speichern:
    info.inputs = std::move(inputs);
    info.labels = std::move(labels);


    return info;
}
