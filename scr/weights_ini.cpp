// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Knoten-Initalisierungsmethoden
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// 
// Version 0.0.3, 25.09.2025
// --------------------------------------

#include "weights_ini.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>

Eigen::MatrixXd initialize_weights(int output_size, int input_size, InitType method) {
    std::random_device rd;
    std::mt19937 gen(rd());

    Eigen::MatrixXd W(output_size, input_size);

    switch (method) {
    case InitType::XAVIER: {
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        break;
    }
    case InitType::HE: {
        double stddev = std::sqrt(2.0 / input_size);
        std::normal_distribution<> dist(0.0, stddev);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        break;
    }
    case InitType::LECUN: {
        double stddev = std::sqrt(1.0 / input_size);
        std::normal_distribution<> dist(0.0, stddev);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        break;
    }
    case InitType::ORTHOGONAL: {
        std::normal_distribution<> dist(0.0, 1.0);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        for (int i = 0; i < output_size; ++i) {
            double norm = W.row(i).norm();
            if (norm > 0)
                W.row(i) /= norm;
        }
        break;
    }
    case InitType::UNIFORM: {
        std::uniform_real_distribution<> dist(-0.05, 0.05);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        break;
    }
    case InitType::NORMAL: {
        std::normal_distribution<> dist(0.0, 0.05);
        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                W(i, j) = dist(gen);
        break;
    }
    case InitType::ITERATIVE: {

        // Placeholder: Load weights from previous training session as basis for current training session
        
        break;
    }
    default:
        throw std::invalid_argument("Unknown initialization method.");
    }

    return W;
}

std::string to_string(InitType init) {
    switch (init) {
    case InitType::HE: return "He";
    case InitType::XAVIER: return "Xavier";
    case InitType::LECUN: return "LeCun";
    case InitType::ORTHOGONAL: return "Orthogonal";
    case InitType::UNIFORM: return "Uniform";
    case InitType::NORMAL: return "Normaldistribution";
	case InitType::ITERATIVE: return "Iterative";
    default: return "Unknown";
    }
}
