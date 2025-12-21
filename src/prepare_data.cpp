// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Trainingsdaten-Preparing-Funktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.5, 14.08.2025
// --------------------------------------


#include "prepare_data.h"
#include "dataset_utils.h" // Für DatasetInfo & Encoding-Funktionen
#include <algorithm>       // Für std::shuffle
#include <random>          // mt19937, random_device
#include <numeric>         // Für std::iota
#include <iostream>        // Für Debug-Ausgaben

// Trainingsdaten vorbereiten:
// Diese Funktion bereitet die Trainings- und Validierungsdaten vor, indem sie die Eingaben
// und Labels in separate Vektoren aufteilt. Sie wird für Klassifikations- und Regressionsprobleme
// verwendet. Bei Klassifikation werden die Labels als One-Hot-Vektoren kodiert,
// bei Regression als 1D-Vektoren mit einem Element. Die Funktion mischt die Daten zufällig
// und teilt sie in Trainings- und Validierungssets auf, basierend auf dem angegebenen
// Anteil der Validierungsdaten (val_split). Die Funktion erwartet eine DatasetInfo-Struktur,
// die die Eingaben und Labels enthält, sowie Vektoren für die Trainings- und Validierungsdaten.
// Die Labels werden je nach Problemtyp unterschiedlich behandelt:
// - Bei Klassifikation: One-Hot-Encoding und Integer-Labels
// - Bei Regression: 1D-Vektoren für Labels und Float-Werte

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
    std::vector<float>& val_labels_float)  {

	// Vorbereitungen: Alle Vektoren leeren:
	train_inputs.clear();
	train_labels_enc.clear();
	val_inputs.clear();
	val_labels_enc.clear();
	train_labels_int.clear();
	val_labels_int.clear();
	train_labels_float.clear();
	val_labels_float.clear();

    // 1) Indizes erzeugen:
    size_t N = ds.inputs.size();
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    // 2) Mischen mit einmaligem RNG
    static std::mt19937 rng{ std::random_device{}() };
    std::shuffle(idx.begin(), idx.end(), rng);


    // 2) Split-Punkt berechnen
    size_t split = static_cast<size_t>(N * (1.0f - val_split));

    // 3) Trainings-Set aufbauen
    for (size_t i = 0; i < split; ++i) {
        size_t j = idx[i];
        train_inputs.push_back(ds.inputs[j]);

        if (ds.is_classification) {
			train_labels_enc.push_back(ds.get_encoded_label(j)); // One-Hot-Encoding
			train_labels_int.push_back(ds.get_label_as_int(j));  // Integer-Label
        }
        else {
            // als 1×1 Eigen-Vektor
            Eigen::VectorXd ev(1);
            ev(0) = ds.get_label_as_float(j);
            train_labels_enc.push_back(ev);

            train_labels_float.push_back(ds.get_label_as_float(j));
        }
    }

    // 4) Validation-Set aufbauen
    for (size_t i = split; i < N; ++i) {
        size_t j = idx[i];
        val_inputs.push_back(ds.inputs[j]);

        if (ds.is_classification) {
            val_labels_enc.push_back(ds.get_encoded_label(j));
            val_labels_int.push_back(ds.get_label_as_int(j));
        }
        else {
            Eigen::VectorXd ev(1);
            ev(0) = ds.get_label_as_float(j);
            val_labels_enc.push_back(ev);

            val_labels_float.push_back(ds.get_label_as_float(j));
        }
    }
}

// Testdaten vorbereiten:
// Diese Funktion bereitet die Testdaten vor, indem sie die Eingaben und Labels
// in separate Vektoren aufteilt. Sie wird für Klassifikations- und Regressionsprobleme
// verwendet. Bei Klassifikation werden die Labels als One-Hot-Vektoren kodiert,
// bei Regression als 1D-Vektoren mit einem Element.

void prepare_test_data(
    const DatasetInfo& ds,
    std::vector<Eigen::VectorXd>& test_inputs,
    std::vector<Eigen::VectorXd>& test_labels_enc,
    std::vector<int>& test_labels_int,       // nur bei Klassifikation
    std::vector<float>& test_labels_float    // nur bei Regression
) {
    test_inputs.clear();
    test_labels_enc.clear();
    test_labels_int.clear();
    test_labels_float.clear();

    for (size_t i = 0; i < ds.inputs.size(); ++i) {
        test_inputs.push_back(ds.inputs[i]);

        if (ds.is_classification) {
            test_labels_enc.push_back(ds.get_encoded_label(i));
            test_labels_int.push_back(ds.get_label_as_int(i));
        }
        else {
            Eigen::VectorXd ev(1);
            ev(0) = ds.get_label_as_float(i);
            test_labels_enc.push_back(ev);
            test_labels_float.push_back(ds.get_label_as_float(i));
        }
    }
}
