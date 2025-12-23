// Project: Config_Utils, Konfiguartionsmanagenemt des NN's
// 
#pragma once
#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include "config.h"         // Deine Config-Struktur
#include <string>

// Speichert die Konfiguration als JSON-Datei
void save_config_as_json(const Config& config, const std::string& filename);

// Lädt die Konfiguration aus einer JSON-Datei
bool load_config_from_json(Config& config, const std::string& filename);

// 
std::string model_type_to_string(ModelType type);

ModelType model_type_from_string(const std::string& str);

std::string opti_type_to_string(OptimizerType opt);

OptimizerType opti_type_from_string(const std::string& str);


#endif // CONFIG_UTILS_H

