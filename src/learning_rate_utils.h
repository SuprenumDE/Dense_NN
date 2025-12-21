// Projekt learning_rate_utils.h
// // Lernraten-Hilfsfunktionen
#pragma once
#ifndef LEARNING_RATE_UTILS_H
#define LEARNING_RATE_UTILS_H

#include <string>

// Funktion zur Anpassung der Lernrate basierend auf dem Lernraten-Modus:
double adjust_learning_rate(const std::string& lr_mode, double base_lr, int epoch);

#endif
