// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Netzqualität-Logging-Funktionen
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.1, 20.07.2025
// --------------------------------------

#include "logging_utils.h"
#include <ctime>
#include <iomanip>
#include <sstream>

// Logging-Funktionen für Trainings- und Validierungsmetriken

std::string timestamp_now() {
    std::time_t now = std::time(nullptr);
    struct tm timeinfo;
    char buf[100];

    // Microsoft-sicher: localtime_s(dst, src)
    if (localtime_s(&timeinfo, &now) == 0) {
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
        return std::string(buf);
    }
    else {
        return "UNKNOWN_TIME";
    }
}


// Schreibt den Header für die Log-Datei:
void write_log_header(std::ofstream& log) {
    log << "# Start of training: " << timestamp_now() << "\n";
    log << "Epoch, Loss, Classification_rate, Val_loss, Val_accuracy, duration, learning_rate\n";
    log.flush();
}

// Loggt eine Epoche mit den entsprechenden Metriken:
void log_epoch(std::ofstream& log,
    int epoch,
    double train_loss,
    double train_accuracy,
    double val_loss,
    double val_accuracy,
    double epoch_time,
    double learning_rate) {

    log << epoch << ","
        << train_loss << ","
        << train_accuracy * 100 << ","
        << val_loss << ","
        << val_accuracy * 100 << ","
        << epoch_time << ","
        << learning_rate << "\n";

    log.flush();
}
