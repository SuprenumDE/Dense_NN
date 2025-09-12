// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// NN-Konfigurationsmanagement, Utilits
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.1, 01.08.2025
// --------------------------------------


#include "config_utils.h"
#include "weights_ini.h" // für to_string() und InitType
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

void save_config_as_json(const Config& config, const std::string& filename) {
    json j = {
        {"modus", config.modus},
        {"dataset_file", config.dataset_file},
        {"architecture", config.architecture},
        {"activations", config.activations},
        {"loss_type", to_string(config.loss_type)},
        {"weights_file", config.weights_file},
        {"epochs", config.epochs},
        {"n_TrainingsSample", config.n_TrainingsSample},
        {"batch_size", config.batch_size},
        {"val_split", config.val_split},
        {"learning_rate", config.learning_rate},
        {"lr_mode", config.lr_mode},
        {"W_init_Methode", to_string(config.W_init_Methode)},
        {"min_delta", config.min_delta},
        {"model_type", model_type_to_string(config.model_type)}
    };

    std::ofstream out(filename);
    out << j.dump(4);
    out.close();
}

bool load_config_from_json(Config& config, const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    json j;
    in >> j;

    config.modus = j["modus"];
    config.dataset_file = j["dataset_file"];
    config.architecture = j["architecture"].get<std::vector<size_t>>();
    config.activations = j["activations"].get<std::vector<std::string>>();
    config.weights_file = j["weights_file"];
    config.epochs = j["epochs"];
    config.n_TrainingsSample = j["n_TrainingsSample"];
    config.batch_size = j["batch_size"];
    config.val_split = j["val_split"];
    config.learning_rate = j["learning_rate"];
    config.lr_mode = j["lr_mode"];
    config.min_delta = j["min_delta"];
    config.model_type = model_type_from_string(j["model_type"]);


    std::string loss_str = j["loss_type"];
    if (loss_str == "MAE") config.loss_type = LossType::MAE;
    else if (loss_str == "MSE") config.loss_type = LossType::MSE;
    else if (loss_str == "CROSS_ENTROPY") config.loss_type = LossType::CROSS_ENTROPY;
    else return false;


    std::string init_str = j["W_init_Methode"];
    if (init_str == "Xavier") config.W_init_Methode = InitType::XAVIER;
    else if (init_str == "He") config.W_init_Methode = InitType::HE;
    else if (init_str == "LeCun") config.W_init_Methode = InitType::LECUN;
    else if (init_str == "Orthogonal") config.W_init_Methode = InitType::ORTHOGONAL;
    else if (init_str == "Uniform") config.W_init_Methode = InitType::UNIFORM;
    else if (init_str == "Normalverteilung") config.W_init_Methode = InitType::NORMAL;
    else return false;

    return true;
}

// ModelType-Funktionen:
std::string model_type_to_string(ModelType type) {
    switch (type) {
    case ModelType::CLASSIFICATION: return "Klassifikation";
    case ModelType::REGRESSION: return "Regression";
    default: return "Unbekannt";
    }
}

ModelType model_type_from_string(const std::string& str) {
    if (str == "Klassifikation") return ModelType::CLASSIFICATION;
    else if (str == "Regression") return ModelType::REGRESSION;
    else throw std::invalid_argument("Unbekannter Modelltyp: " + str);
}

