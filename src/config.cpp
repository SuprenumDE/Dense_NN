// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// NN-Konfigurationsmanagement
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.3, 13.11.2025
// --------------------------------------


#include "config.h"
#include <iostream>
#include "weights_ini.h" // für to_string()


void print_config(const Config& config, std::ostream& out) {
    out << "\nCurrent configuration:\n";
    out << "Mode: " << config.modus << "\n";
    out << "Data set: " << config.dataset_file << "\n";
    out << "Architecture:: ";
    for (auto size : config.architecture) out << size << " ";
    out << "\nActivations: ";
    for (const auto& act : config.activations) out << act << " ";
    out << "\nWeights file: " << config.weights_file << "\n";
    out << "Epochs: " << config.epochs << "\n";
    out << "Training samples: " << config.n_TrainingsSample << "\n";
    out << "Batch size: " << config.batch_size << "\n";
    out << "Validation rate: " << config.val_split << "\n";
    out << "Learning rate: " << config.learning_rate << "\n";
    out << "LR mode: " << config.lr_mode << "\n";
    out << "Init method: " << to_string(config.W_init_Methode) << "\n";
	out << "Loss function: " << to_string(config.loss_type) << "\n";
    out << "Optimizer: " << to_string(config.optimizer_type) << "\n";
    out << "Minimal improvement: " << config.min_delta << "\n\n";
}
