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
#include <fstream>

std::string timestamp_now() {
    std::time_t now = std::time(nullptr);
    std::tm timeinfo{};
    char buf[100];

// Which OS?
#if defined(_WIN32)
    // Windows: localtime_s(&dst, &src)
    if (localtime_s(&timeinfo, &now) == 0)
#else
    // POSIX (Linux, macOS, Raspberry Pi): localtime_r(&src, &dst)
    if (localtime_r(&now, &timeinfo) != nullptr)
#endif
    {
        // Variante 1:
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
        return std::string(buf);

        // Variante 2:
        /*
        std::ostringstream oss;
        oss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
        return oss.str();
        */
    }

    return "UNKNOWN_TIME";
}

void write_log_header(std::ofstream& log) {
    log << "# Start of training: " << timestamp_now() << "\n";
    log << "Epoch, Loss, Classification_rate, Val_loss, Val_accuracy, duration, learning_rate\n";
    log.flush();
}

void log_epoch(std::ofstream& log,
    int epoch,
    double train_loss,
    double train_accuracy,
    double val_loss,
    double val_accuracy,
    double epoch_time,
    double learning_rate)
{
    log << epoch << ","
        << train_loss << ","
        << train_accuracy * 100 << ","
        << val_loss << ","
        << val_accuracy * 100 << ","
        << epoch_time << ","
        << learning_rate << "\n";

    log.flush();
}