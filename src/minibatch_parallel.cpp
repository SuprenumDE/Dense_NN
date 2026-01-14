// 
// Dense neural network based on “Eigen”
// 
// NN-Mini-batch training
// (Moved to avoid conflicts with cxxopts.hpp caused by omp.hpp)
// 
// Developer: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
//
// Version 0.0.1, 10.01.2026
// --------------------------------------

#define EIGEN_USE_THREADS           // Activate OpneMP for Eigen
#define EIGEN_USE_OPENMP            // Activate OpneMP for Eigen
#include "Eigen/Eigen"
#include <omp.h>
#include <thread>

#include "minibatch_parallel.h"
#include "dataset_utils.h"
#include "loss_functions.h"
#include "activation_functions.h"
#include "metrics_utils.h"
#include <iostream>

// Thread-Info
static void print_eigen_thread_info_once() {
    static bool printed = false;
    if (!printed) {
        std::cout << "Eigen parallel backend active: " << Eigen::nbThreads()
            << " threads\n"; std::cout
            << "OpenMP max threads: " << omp_get_max_threads()
            << "\n"; std::cout << "CPU cores available: "
            << std::thread::hardware_concurrency() << "\n\n"; printed = true;
    }
}

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
)
{
    // Parallel area:

    print_eigen_thread_info_once();

#pragma omp parallel
    {
        // Thread-local accumulators:
        std::vector<Eigen::MatrixXd> dw_private(weights.size());
        std::vector<Eigen::VectorXd> db_private(biases.size());

        for (size_t l = 0; l < weights.size(); ++l) {
            dw_private[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
            db_private[l] = Eigen::VectorXd::Zero(biases[l].size());
        }

        double local_loss = 0.0;
        int local_correct = 0;

        // Parallel loop over samples:
        int j_begin = static_cast<int>(start_index);
        int j_end = static_cast<int>(end_index);

#pragma omp for nowait
        for (int j = j_begin; j < j_end; ++j) {

            // Forward-Pass
            std::vector<Eigen::VectorXd> a_values;
            std::vector<Eigen::VectorXd> z_values;

            Eigen::VectorXd a = train_inputs[j];
            a_values.push_back(a);

            for (size_t l = 0; l < weights.size(); ++l) {
                Eigen::VectorXd z = weights[l] * a + biases[l];
                z_values.push_back(z);

                auto act = Activation::get(activations[l]);
                a = act->activate(z);
                a_values.push_back(a);
            }

            // Backward-Pass
            std::vector<Eigen::VectorXd> deltas(weights.size());

            Eigen::VectorXd prediction = a_values.back();
            Eigen::VectorXd target = train_labels_enc[j];

            Eigen::VectorXd dz = loss_fn->gradient(prediction, target);
            deltas.back() = dz;

            for (int l = static_cast<int>(weights.size()) - 2; l >= 0; --l) {
                Eigen::VectorXd da = weights[l + 1].transpose() * deltas[l + 1];
                auto act = Activation::get(activations[l]);
                Eigen::VectorXd dz_l = da.array() * act->derivative(z_values[l]).array();
                deltas[l] = dz_l;
            }

            // Thread-local gradients:
            for (size_t l = 0; l < weights.size(); ++l) {
                dw_private[l] += deltas[l] * a_values[l].transpose();
                db_private[l] += deltas[l];
            }

            // Loss
            local_loss += loss_fn->compute(prediction, target);

            // Accuracy
            if (is_classification) {
                int y_true = train_labels_int[j];
                if (argmax(prediction) == y_true)
                    local_correct++;
            }
            else {
                double y_true = train_labels_float[j];
                if (std::abs(prediction(0) - y_true) < tolerance)
                    local_correct++;
            }
        }

        // Reduction:
#pragma omp critical
        {
            for (size_t l = 0; l < weights.size(); ++l) {
                dw_acc[l] += dw_private[l];
                db_acc[l] += db_private[l];
            }

            epoch_loss += local_loss;
            correct_predictions += local_correct;
        }
    }
}
