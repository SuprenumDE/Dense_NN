#pragma once
#ifndef WEIGHTS_IO_H
#define WEIGHTS_IO_H

#include <string>
#include <vector>
#include "Eigen/Eigen"


// Speichern: beliebig viele Layer im Unterverzeichnis <prefix>
void save_all_weights_biases_csv(
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const std::string& prefix = "weights");

// Laden: automatisch erkannte Layeranzahl aus Unterverzeichnis <prefix>
bool load_all_weights_biases_csv(
    std::vector<Eigen::MatrixXd>& weights,
    std::vector<Eigen::VectorXd>& biases,
    const std::string& prefix = "weights");

#endif // WEIGHTS_IO_H



