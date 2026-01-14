#pragma once
#ifndef MINIBATCH_PARALLEL_H
#define MINIBATCH_PARALLEL_H

#include <vector>
#include "Eigen/Dense"

struct Dataset;
class LossFunction;
class Activation;

// Performs training on a mini-batch.
// The gradients are accumulated in dw_acc and db_acc.
void train_minibatch(
    size_t start_index,
    size_t end_index,
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const std::vector<Eigen::VectorXd>& train_inputs,
    const std::vector<Eigen::VectorXd>& train_labels_enc,
    const std::vector<int>& train_labels_int,
    const std::vector<float>& train_labels_float,
    bool is_classification,
    double tolerance,
    LossFunction* loss_fn,
    const std::vector<std::string>& activations,
    std::vector<Eigen::MatrixXd>& dw_acc,
    std::vector<Eigen::VectorXd>& db_acc,
    double& epoch_loss,
    double& correct_predictions
);

#endif // MINIBATCH_PARALLEL_H
