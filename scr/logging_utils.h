#pragma once
#ifndef LOGGING_UTILS_H
#define LOGGING_UTILS_H

#include <fstream>
#include <string>

// Header-Datei für Logging-Funktionen:
void write_log_header(std::ofstream& log);

// Loggt eine Epoche mit den entsprechenden Metriken:
void log_epoch(std::ofstream& log,
    int epoch,
    double train_loss,
    double train_accuracy,
    double val_loss,
    double val_accuracy,
    double epoch_time,
    double learning_rate);

#endif // LOGGING_UTILS_H