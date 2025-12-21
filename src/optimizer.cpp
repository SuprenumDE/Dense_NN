// NN optimizer functions
// 
// Dense neural network based on “Eigen”
// 
// Developer: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// Lizenz: MIT (also for all self-developed included h and cpp files)
// Version 0.0.1, 11.11.2025
// --------------------------------------

#include "optimizer.h"

// ------------------- Parsing -------------------
OptimizerType parse_optimizer(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    if (t == "sgd")     return OptimizerType::SGD;
    if (t == "rmsprop") return OptimizerType::RMSProp;
    if (t == "adam")    return OptimizerType::Adam;
    throw std::invalid_argument("Unknown optimizer: " + s);
}

// ------------------- Initialization -------------------
void init_optimizer_state(
    OptimizerState& state,
    const std::vector<Eigen::MatrixXd>& weights,
    const std::vector<Eigen::VectorXd>& biases,
    const OptimizerParams&)  
{
    const size_t L = weights.size();
    if (biases.size() != L) {
        throw std::invalid_argument("init_optimizer_state: weights/biases size mismatch");
    }

    // RMSProp: G (accumulator of squared grads)
    state.G_W.resize(L);
    state.G_b.resize(L);
    for (size_t l = 0; l < L; ++l) {
        state.G_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
        state.G_b[l] = Eigen::VectorXd::Zero(biases[l].size());
    }

    // Adam: first moment (M) and second moment (V)
    state.M_W.resize(L);
    state.M_b.resize(L);
    state.V_W.resize(L);
    state.V_b.resize(L);
    for (size_t l = 0; l < L; ++l) {
        state.M_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
        state.M_b[l] = Eigen::VectorXd::Zero(biases[l].size());
        state.V_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
        state.V_b[l] = Eigen::VectorXd::Zero(biases[l].size());
    }

    // Adam-Zähler
    state.t = 0;
}

// ------------------- Update -------------------
void apply_update(
    std::vector<Eigen::MatrixXd>& weights,
    std::vector<Eigen::VectorXd>& biases,
    const std::vector<Eigen::MatrixXd>& dW,
    const std::vector<Eigen::VectorXd>& db,
    OptimizerState& state,
    const OptimizerParams& params,
    double learning_rate)
{
    const size_t L = weights.size();
    if (biases.size() != L || dW.size() != L || db.size() != L) {
        throw std::invalid_argument("apply_update: size mismatch among layers");
    }

    // Gemeinsame Konstanten
    const double eps = params.epsilon;

    switch (params.type) {
    case OptimizerType::SGD: {
        // Klassisches SGD: w -= lr * dW; b -= lr * db
        for (size_t l = 0; l < L; ++l) {
            weights[l].noalias() -= learning_rate * dW[l];
            biases[l].noalias() -= learning_rate * db[l];
        }
        break;
    }

    case OptimizerType::RMSProp: {
        // G = rho * G + (1 - rho) * d^2
        const double rho = params.rho;
        const double one_minus_rho = 1.0 - rho;

        // Sicherstellen, dass State initialisiert ist
        if (state.G_W.size() != L || state.G_b.size() != L) {
            throw std::logic_error("RMSProp: state not initialized. Call init_optimizer_state first.");
        }

        for (size_t l = 0; l < L; ++l) {
            // Elementweise Quadrate
            state.G_W[l] = rho * state.G_W[l].array() + one_minus_rho * dW[l].array().square();
            state.G_b[l] = rho * state.G_b[l].array() + one_minus_rho * db[l].array().square();

            // Update: w -= lr * d / (sqrt(G) + eps)
            weights[l].array() -= learning_rate * dW[l].array() / (state.G_W[l].array().sqrt() + eps);
            biases[l].array() -= learning_rate * db[l].array() / (state.G_b[l].array().sqrt() + eps);
        }
        break;
    }

    case OptimizerType::Adam: {
        // t-Inkrement
        state.t += 1;
        const double b1 = params.beta1;
        const double b2 = params.beta2;

        if (state.M_W.size() != L || state.V_W.size() != L ||
            state.M_b.size() != L || state.V_b.size() != L) {
            throw std::logic_error("Adam: state not initialized. Call init_optimizer_state first.");
        }

        // Bias Correction Faktoren
        const double b1_t = std::pow(b1, static_cast<double>(state.t));
        const double b2_t = std::pow(b2, static_cast<double>(state.t));
        const double corr1 = 1.0 - b1_t;
        const double corr2 = 1.0 - b2_t;

        for (size_t l = 0; l < L; ++l) {
            // Erste Momente (M) und zweite Momente (V)
            state.M_W[l] = b1 * state.M_W[l].array() + (1.0 - b1) * dW[l].array();
            state.M_b[l] = b1 * state.M_b[l].array() + (1.0 - b1) * db[l].array();

            state.V_W[l] = b2 * state.V_W[l].array() + (1.0 - b2) * dW[l].array().square();
            state.V_b[l] = b2 * state.V_b[l].array() + (1.0 - b2) * db[l].array().square();

            // Bias-korrigierte Schätzungen
            Eigen::MatrixXd mhat_W = state.M_W[l].array() / corr1;
            Eigen::VectorXd mhat_b = state.M_b[l].array() / corr1;

            Eigen::MatrixXd vhat_W = state.V_W[l].array() / corr2;
            Eigen::VectorXd vhat_b = state.V_b[l].array() / corr2;

            // Update
            weights[l].array() -= learning_rate * mhat_W.array() / (vhat_W.array().sqrt() + eps);
            biases[l].array() -= learning_rate * mhat_b.array() / (vhat_b.array().sqrt() + eps);
        }
        break;
    }

    default:
        throw std::invalid_argument("apply_update: unknown optimizer type");
    }
}


