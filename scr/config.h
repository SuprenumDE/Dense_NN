// Konfigurationsmanagement für das neuronale Netzwerk:

#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include "weights_ini.h"           // Netzgewichtsinitalisierung
#include "loss_functions.h"        // Loss-Funktionstypen

enum class ModelType { CLASSIFICATION, REGRESSION };

struct Config {
    int modus = 1;
    std::string dataset_file;
    std::vector<size_t> architecture;
    std::vector<std::string> activations;
    std::string weights_file = "weights.txt";
    int epochs = 100;
    int n_TrainingsSample = 1000; 
    int batch_size = 64;
    float val_split = 0.2f;
    double learning_rate = 0.01;
    std::string lr_mode;
    InitType W_init_Methode = InitType::HE;
    double min_delta = 0.0001;
    LossType loss_type = LossType::MAE;
    ModelType model_type = ModelType::CLASSIFICATION;
};

// Konfiguration ausgeben:
void print_config(const Config& config, std::ostream& out);


#endif // CONFIG_H

