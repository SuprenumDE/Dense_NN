// train_nn.cpp /Dense_NN
// Dense neural network based on “Eigen”
// 
// Developer: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// Lizenz: MIT (also for all self-developed included h and cpp files)
// Version 0.1.30, 10.01.2026
// Eigen-Version: 3.4.0
// C-Version: ISO-Standard C++20
// --------------------------------------

#pragma warning(push)
#pragma warning(disable: 5030)      // Disables selectany-related warnings
#include "cxxopts.hpp"              // Argument parser
#pragma warning(pop)         

#include "Eigen/Eigen"
#include "weights_io.h"             // Load/Save Weights and Bias
#include "activation_functions.h"   // Activation functions
#include "loss_functions.h"         // Verlustfunktionen
#include "weights_ini.h"            // Weight initialization
#include "learning_rate_utils.h"    // Learning rate support functions
#include "dataset_utils.h"          // Data handling functions
#include "logging_utils.h"          // Logging functions
#include "prepare_data.h"           // Prepare data for training/validation
#include "config.h"                 // Neural network configuration        
#include "config_utils.h"           // Configuration management
#include "diagnostics_utils.h"      // Diagnostic functions for training
#include "metrics_utils.h"          // Metrics for classification/regression (helper functions)
#include "optimizer.h"              // Optimizer functions
#include "minibatch_parallel.h"     // Minibatch-Training
#include "resource.h"               // Icon ...


#include <iostream>
#include <fstream>
#include <sstream>                  // String stream for CSV lines
#include <vector>
#include <cmath>                    
#include <random>
#include <iomanip>                  // Formatted output
#include <chrono>                   
#include <stdexcept>                
#include <set>                      // Set for unique labels


using namespace std;

// Parse arguments and store in config object:
std::optional<Config> parse_arguments(const cxxopts::ParseResult& result) {
    try {
        Config config;

       
        config.modus = result["mode"].as<int>();

        // Architektur parsen
        if (config.modus == 1) {

            if (!result.count("dataset")) {
                std::cerr << "Error: No training data set specified via -f. Example: -f filename.csv\n";
                return std::nullopt;
            }
            config.dataset_file = result["dataset"].as<std::string>();

            std::stringstream ss_arch(result["architecture"].as<std::string>());
            std::string layer;
            while (std::getline(ss_arch, layer, ',')) {
                config.architecture.push_back(std::stoul(layer));
            }

            if (config.architecture.empty()) {
                std::cerr << "Error: Architecture must not be empty. Example: -a 64,32,10\n";
                return std::nullopt;
            }

            // Parse activation functions:
            std::stringstream ss_act(result["activations"].as<std::string>());
            std::string act;
            while (std::getline(ss_act, act, ',')) {
                config.activations.push_back(trim(act));
            }
            if (config.activations.size() != config.architecture.size() - 1) {
                std::cerr << "Error: The number of activations must be equal to the number of layer transitions.\n";
                return std::nullopt;
            }

            // Optimizer Gradient Descent Method Parse:
            std::string opt_str = result["optimizer"].as<std::string>();
            try {
                config.optimizer_type = parse_optimizer(opt_str);
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Error: Unknown optimizer \"" << opt_str
                    << "\", default value (SGD) is used.\n";
                config.optimizer_type = OptimizerType::SGD;
            }
            // Optimizer parameters:
            config.optimizer_params.type = config.optimizer_type;
            config.optimizer_params.beta1 = result["beta1"].as<double>();
            config.optimizer_params.beta2 = result["beta2"].as<double>();
            config.optimizer_params.epsilon = result["epsilon"].as<double>();
            config.optimizer_params.rho = result["rho"].as<double>();

            // Further Parameters:
            config.weights_file = result["weights"].as<std::string>();
            config.epochs = result["epochs"].as<int>();
            config.n_TrainingsSample = result["samples"].as<int>();
            config.batch_size = result["batch"].as<int>();
            config.val_split = result["val"].as<float>();
            config.learning_rate = result["learning_rate"].as<double>();
            config.lr_mode = result["lr_mode"].as<std::string>();
            config.min_delta = result["min_delta"].as<double>();

            // Parse Loss Function:
            std::string loss = result["loss"].as<std::string>();
            if (loss == "CROSS_ENTROPY")
                config.loss_type = LossType::CROSS_ENTROPY;
            else if (loss == "MSE")
                config.loss_type = LossType::MSE;
            else if (loss == "MAE")
                config.loss_type = LossType::MAE;
            else {
                std::cerr << "Error: Unknown loss function \"" << loss << "\", default value (CROSS_ENTROPY) is used.\n";
                config.loss_type = LossType::CROSS_ENTROPY;
            }

            // Parse initialization method:
            std::string init = result["init"].as<std::string>();
            if (init == "HE") config.W_init_Methode = InitType::HE;
            else if (init == "XAVIER") config.W_init_Methode = InitType::XAVIER;
            else if (init == "LECUN") config.W_init_Methode = InitType::LECUN;
            else if (init == "ORTHOGONAL") config.W_init_Methode = InitType::ORTHOGONAL;
            else if (init == "UNIFORM") config.W_init_Methode = InitType::UNIFORM;
            else if (init == "NORMAL") config.W_init_Methode = InitType::NORMAL;
            else if (init == "ITERATIVE") config.W_init_Methode = InitType::ITERATIVE;
            else {
                std::cerr << "Error: Unknown initialization method \"" << init << "\", default value (HE) is used.\n";
                config.W_init_Methode = InitType::HE;
            }

            // Check mini batch size:
            if (config.batch_size != 32 && config.batch_size != 64 && config.batch_size != 128) {
                std::cerr << "Error: Invalid batch size: " << config.batch_size << ". Only 32, 64, or 128 are allowed.\n";
                return std::nullopt;
            }

            if (config.val_split <= 0.0f || config.val_split >= 1.0f) {
                std::cerr << "Error: Validation percentage must be between 0 and 1.\n";
                return std::nullopt;
            }

        }

        else if (config.modus == 2) {
            
            if (!result.count("dataset")) {
                std::cerr << "Error: No test data set specified via -f. Example: -f filename.csv\n";
                return std::nullopt;
            }

            config.dataset_file = result["dataset"].as<std::string>();

           
		}

        return config;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: Error parsing argument: " << e.what() << std::endl;
        return std::nullopt;
    }
}


// Hilfsfunktion, um die Hilfe anzuzeigen:
void show_help(const cxxopts::Options& options) {
    std::cout << R"(
  _________.__                 __________        __   
 /   _____/|  |__   ____      \______   \ _____/  |_ 
 \_____  \ |  |  \_/ __ \      |       _// __ \   __\
 /        \|   Y  \  ___/      |    |   \  ___/|  |  
/_______  /|___|  /\___  >     |____|_  /\___  >__|  
        \/      \/     \/             \/     \/      

EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.30, 10.01.2026, License: MIT
)";
    std::cout << "\nUsage:\n";
    std::cout << "  ./Dense_NN [Optionen]\n\n";
    std::cout << "Available options:\n";
    std::cout << options.help({ "" }) << "\n";
    std::cout << "Example:\n";
    std::cout << "  ./Dense_NN --architecture=784,128,64,10 --activations=relu,relu,softmax --epochs=100 --lr=0.01 --optimizer=SGD --dataset=mnist.csv\n\n";
    std::cout << "Further information:\n";
    std::cout << "  Guenter Faes, Mail: eigennet@faes.de\n";
    std::cout << "  YouTube: https://www.youtube.com/@r-statistik\n";
    std::cout << "  GitHub:  https://github.com/SuprenumDE/EigenNET\n";
}

// ------------------------------- Ende Hilfsfunktionen -----------------





int main(int argc, char* argv[]) {

    // ----------------- Initialisierung -----------------
	vector<size_t> architecture;                        // Layer sizes, e.g. 784,128,64,10
	vector<std::string> activations;                    // Activation functions, z. B. relu,relu,tanh,softmax
    int modus = 1;                                      // Processing mode: 1=Training, 2=Testing, 3=Exit
    int epochs = 0;                                     // Training epoch, 0 = Default
    int n_TrainingsSample = 1000;                       // To load just a few training data
    size_t batch_size = 64;                             // Minibatch size, default value
    float val_split = 0.2f;                             // Proportion of data for validation (e.g., 0.2)
    double learning_rate = 0.01;                        // Learning rate, default value
    string lr_mode = {};                                // Dynamic learning rate mode, e.g., “decay” or “step”
    InitType W_init_Methode = InitType::HE;             // Initialization method for weights, weights_ini.h
    LossType loss_type = LossType::CROSS_ENTROPY;       // Loss function, e.g., “CROSS_ENTROPY,” “MAE,” or “MSE”
    OptimizerType optimizer_type = OptimizerType::SGD;  // Optimizer: "SGD", "RMSProp", "Adam"

    // Kontrollvariablen
    double correct_predictions = 0;
    double accuracy = 0.0;
    double epoch_loss = 0.0;
    double val_loss = 0.0;
    double val_correct = 0;
    double val_accuracy = 0.0;
	double tolerance = 0.01;                            // Tolerance for accuracy, default value 0.01

    //Variablen für frühes Stoppen des Trainings:
	double best_val_loss = 0.0;                         // Best validation loss
    int patience_counter = 0;                           // Patience counter
    const int patience_limit = 5;                       // For example, 5 eras of patience
    double min_delta = 1e-4;                            // minimal improvement

    // Dateihandling:
    string weights_file = "weights.txt";                // File name for weights and bias
    string dataset_file = "mnist.csv";                  // Example specification: Standard MNIST dataset

	// Datenstrukturen Dataset-Utils
	DatasetInfo dataset; 				                // Structure for data set information

	// Neuronale Netz-Parameter: 
	vector<Eigen::MatrixXd> weights;                    // Weights of the layers
	vector<Eigen::VectorXd> biases;                     // Layers biases
	vector<Eigen::VectorXd> a_values;                   // Activation values of the layers
	vector<Eigen::VectorXd> z_values;                   // Z values of the layers (pre-activation)
	long long total_parameters = 0;                     // Total number of parameters in the network

    // -------------------------------------------------------

    
        cxxopts::Options options("Dense_NN", "Train or test a neural network");

        options.add_options()
            ("m,mode", "Mode: 1=Train, 2=Test, 3=Exit", cxxopts::value<int>()->default_value("1"))
            ("f,dataset", "Data set file (e.g., filename.csv)", cxxopts::value<std::string>())
            ("c,architecture", "Layer sizes separated by commas, e.g. 784,128,64,10", cxxopts::value<std::string>())
            ("x,activations", "Activation functions per layer, separated by commas, e.g., relu, relu, tanh, softmax", cxxopts::value<std::string>())
            ("w,weights", "Weights file (e.g., weights.txt)", cxxopts::value<std::string>()->default_value("weights.txt"))
            ("e,epochs", "Number of epochs", cxxopts::value<int>()->default_value("100"))
            ("z,loss", "Loss function: CROSS_ENTROPY, MAE, or MSE", cxxopts::value<std::string>()->default_value("CROSS_ENTROPY"))
            ("s,samples", "Number of training samples", cxxopts::value<int>()->default_value("1000"))
            ("b,batch", "Minibatch size (32, 64, or 128)", cxxopts::value<int>()->default_value("64"))
            ("v,val", "Proportion of data for validation (e.g., 0.2)", cxxopts::value<float>()->default_value("0.2"))
            ("l,learning_rate", "Learning rate", cxxopts::value<double>()->default_value("0.01"))
            ("r,lr_mode", "Learning rate mode (e.g., decay, step)", cxxopts::value<std::string>()->default_value(""))
            ("i,init", "Initialization method (HE, XAVIER, ...)", cxxopts::value<std::string>()->default_value("HE"))
            ("d,min_delta", "Minimal improvement for early training stop", cxxopts::value<double>()->default_value("0.0001"))

            ("o,optimizer", "Optimizer: SGD, RMSProp, Adam", cxxopts::value<std::string>()->default_value("SGD"))
            ("beta1", "Adam: first moment decay (beta1)", cxxopts::value<double>()->default_value("0.9"))
            ("beta2", "Adam: second moment decay (beta2)", cxxopts::value<double>()->default_value("0.999"))
            ("eps,epsilon", "Stability term (Adam/RMSProp)", cxxopts::value<double>()->default_value("1e-8"))
            ("rho", "RMSProp: Decay-Rate for squared gradients", cxxopts::value<double>()->default_value("0.9"))

			("p,print_config", "Outputs the current configuration")
            ("h,help", "Displays help");
        

	//  Evaluate Argument Options:
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            show_help(options);
            return 0;
        }

        int mode = result["mode"].as<int>();
        if (mode < 1 || mode > 3) {
            std::cerr << "Invalid mode: " << mode << ". Only 1 (Train), 2 (Test), or 3 (Exit) are allowed.\n";
            return 1;
		}

        auto config_opt = parse_arguments(result);
        Config config = *config_opt;  // Configuration from the arguments

        if (result.count("print_config")) {
            print_config(config, std::cout);
        }
       

        if (config.modus == 1) {

            std::cout << "EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.30, 10.01.2026\n";
            std::cout << "Training mode activated.\n\n";

            // ---------- Training mode ---------------------

             // Transfer configuration to the variables actually used:
            dataset_file = config.dataset_file;
            architecture = config.architecture;
            activations = config.activations;
            weights_file = config.weights_file;
            epochs = config.epochs;
            n_TrainingsSample = config.n_TrainingsSample;
            batch_size = config.batch_size;
            val_split = config.val_split;
            lr_mode = config.lr_mode;
            W_init_Methode = config.W_init_Methode;
            min_delta = config.min_delta;
            loss_type = config.loss_type;
            optimizer_type = config.optimizer_type;



            // Load data and automatically determine dimensions for input and output layers:
            DatasetInfo dataset = load_dataset_info(dataset_file, n_TrainingsSample);
            report_dataset_stats(dataset, val_split); // Output of the statistics of the data set

            // Validate activation functions:
            for (const auto& act_name : activations) {
                try {
                    Activation::get(act_name); // Test retrieval
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Incorrect activation function detected: \"" << e.what() << "\"\n";
                    std::cerr << "Available options are: relu, sigmoid, tanh, softmax, none\n";
                    return EXIT_FAILURE;
                }
            }


            // Create loss function:
            unique_ptr<LossFunction> loss_fn = createLossFunction(loss_type);

            // Initialize weights and bias, since input_dim is now known:

            std::mt19937 gen(std::random_device{}());            // Random number generator for bias initialization
            std::normal_distribution<> bias_dist(0.0, 0.01);     // Mean value 0, standard deviation 0.01

            // Ensure that the architecture is compatible with the data:
            if (architecture.front() != dataset.input_dim) {
                architecture[0] = dataset.input_dim;            // Adjust input layer size
                std::cout << "Warning: Input layer size adjusted to match dataset input dimension: " << dataset.input_dim << "\n\n";
            }

            // Initialization of weights:

            // Check whether iterative training, load weights from previous training:

            std::vector<Eigen::MatrixXd> pretrained_weights;
            std::vector<Eigen::VectorXd> pretrained_biases;

            if (W_init_Methode == InitType::ITERATIVE) {

                std::cout << "Iterative training selected. Attempting to load pre-trained weights...\n\n";

                // Loading the net weights from the previous training session:
                string weight_prefix = "weights";            // z. B. run42          TODO: Subdirectory for weights hard-wired, needs to be revised!
                if (!load_all_weights_biases_csv(pretrained_weights, pretrained_biases, weight_prefix)) {
                    throw std::runtime_error("Iterative initialization failed: Could not load weights.");
                }
            } // End of iterative weight loading

            for (size_t i = 0; i < architecture.size() - 1; ++i) {
                size_t input_dim = architecture[i];
                size_t output_dim = architecture[i + 1];

                if (W_init_Methode == InitType::ITERATIVE) {
                    // Assign pre-trained weights:
                    if (i >= pretrained_weights.size())
                        throw std::runtime_error("Missing pretrained weights for layer " + std::to_string(i));

                    weights.push_back(pretrained_weights[i]);
                    biases.push_back(pretrained_biases[i]);

                }
                else {
                    // Normally: Initialize weights and bias:
                    // This is where the weights are initialized:
                    Eigen::MatrixXd W = initialize_weights(output_dim, input_dim, W_init_Methode);
                    weights.push_back(W);

                    // This is where the biases are initialized:
                    Eigen::VectorXd b(output_dim);                     // Bias vector for the layers
                    for (int i = 0; i < output_dim; ++i)
                        b(i) = bias_dist(gen);

                    // Add biases:
                    biases.push_back(b);
                }

            }

            // Output layer and number of parameters:
            for (size_t i = 0; i < weights.size(); ++i) {
                std::cout << "Layer " << i << ": "
                    << weights[i].rows() << "x" << weights[i].cols() << " = " << weights[i].rows() * weights[i].cols() << " parameters\n";
                total_parameters += weights[i].rows() * weights[i].cols();
            }
            std::cout << "---------------------------------------\n" << "Total number of parameters: " << total_parameters << "\n\n";
            std::cout << flush;

            val_split = std::clamp(val_split, 0.0f, 1.0f);     // Ensure that val_split is within the valid range

            size_t num_classes = dataset.num_classes;          // Number of classes, only relevant for classification


            // Shuffle & split in training and validation data:
            vector<Eigen::VectorXd> train_inputs;
            vector<Eigen::VectorXd> train_labels_enc;
            vector<Eigen::VectorXd> val_inputs;
            vector<Eigen::VectorXd> val_labels_enc;
            // Classification labels as integers or floats:
            vector<int> train_labels_int;
            vector<int> val_labels_int;
            vector<float> train_labels_float;
            vector<float> val_labels_float;

            prepare_training_data(dataset, val_split,
                train_inputs, train_labels_enc,
                val_inputs, val_labels_enc,
                train_labels_int, val_labels_int,
                train_labels_float, val_labels_float);


            // Tolerance calculation:
            if (dataset.is_classification) {
                config.model_type = ModelType::CLASSIFICATION;
                // For classification: Tolerance based on the number of classes:
                tolerance = 0.1 * num_classes;  // z.B. für 10 Klassen, Toleranz = 1.0
            }
            else {
                // For regression: tolerance based on the standard deviation of the labels:
                if (train_labels_float.empty()) {
                    cerr << "Error: No training labels found for regression." << endl;
                    return 1;
                }
                else {
                    config.model_type = ModelType::REGRESSION;
                    // Calculation of the standard deviation of the training labels:
                    double mean = std::accumulate(train_labels_float.begin(), train_labels_float.end(), 0.0) / train_labels_float.size();
                    double sq_sum = std::inner_product(train_labels_float.begin(), train_labels_float.end(), train_labels_float.begin(), 0.0);
                    double stdev = std::sqrt(sq_sum / train_labels_float.size() - mean * mean);
                    tolerance = 0.1 * stdev; // e.g., 10% of the standard deviation
                }
            }

            // Initializations for the optimizer methods (RMSProp / Adam / ...):
            OptimizerState state;
            state.t = 0;
            state.M_W.resize(weights.size());
            state.V_W.resize(weights.size());
            state.M_b.resize(biases.size());
            state.V_b.resize(biases.size());
            state.G_W.resize(weights.size());
            state.G_b.resize(biases.size());

            for (size_t l = 0; l < weights.size(); ++l) {
                state.M_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
                state.V_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
                state.M_b[l] = Eigen::VectorXd::Zero(biases[l].size());
                state.V_b[l] = Eigen::VectorXd::Zero(biases[l].size());
                state.G_W[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
                state.G_b[l] = Eigen::VectorXd::Zero(biases[l].size());
            }


            // ------------------ Start training ------------------------------------

            // Time measurement for the entire training session:
            auto training_start_time = chrono::high_resolution_clock::now();

            // Create logbook:
            ofstream log("training_log.csv");
            if (!log.is_open()) {
                cerr << "Error: Log file could not be opened." << endl;
                return 1;
            }
            // Save network parameters as a JSON file:
            save_config_as_json(config, "nn_parameter.json");


            // Write logbook header:
            write_log_header(log);

            // Training Loop:
            for (int epoch = 0; epoch < epochs; ++epoch) {

                // Time measurement for the epoch:
                auto start_time = chrono::high_resolution_clock::now();

                epoch_loss = 0.0;
                correct_predictions = 0;
                val_loss = 0.0;
                val_correct = 0;


                // Batch-Training: 
                for (size_t i = 0; i < train_inputs.size(); i += batch_size) {

                    size_t end = std::min(i + batch_size, train_inputs.size());

                    // Initialization of activations and intermediate results:
                    vector<Eigen::MatrixXd> dw_acc(weights.size());
                    vector<Eigen::VectorXd> db_acc(biases.size());
                    for (size_t l = 0; l < weights.size(); ++l) {
                        dw_acc[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
                        db_acc[l] = Eigen::VectorXd::Zero(biases[l].size());
                    }


                    // MINIBATCH-Training:
                    train_minibatch(
                        i,
                        end,
                        weights,
                        biases,
                        train_inputs,
                        train_labels_enc,
                        train_labels_int,
                        train_labels_float,
                        dataset.is_classification,
                        tolerance,
                        loss_fn.get(),
                        activations,
                        dw_acc,
                        db_acc,
                        epoch_loss,
                        correct_predictions
                    );

 

                    for (size_t l = 0; l < weights.size(); ++l) {
                        dw_acc[l] /= static_cast<double>(end - i); // Average value over batch
                        db_acc[l] /= static_cast<double>(end - i);
                    }

                // Adjust learning rate:
                config.learning_rate = adjust_learning_rate(lr_mode, config.learning_rate, epoch);

                // Optimizer-Update:
                apply_update(weights, biases,
                    dw_acc, db_acc,
                    state,
                    config.optimizer_params,
                    config.learning_rate);



                } // End of batch training loop

                // Calculate validation loss and accuracy:
                for (size_t i = 0; i < val_inputs.size(); ++i) {

                    Eigen::VectorXd a = val_inputs[i];

                    for (size_t l = 0; l < weights.size(); ++l) {
                    Eigen::VectorXd z = weights[l] * a + biases[l];
                    auto act = Activation::get(activations[l]);
                    a = act->activate(z);
                    }

                Eigen::VectorXd target = val_labels_enc[i];

                // Calculate validation loss:
                val_loss += loss_fn->compute(a, target);

                // Counting correct classifications:
                if (dataset.is_classification) {
                    int y_true = val_labels_int[i];
                    if (argmax(a) == y_true) ++val_correct;
                }
                else {
                    double y_true = train_labels_float[i];
                    if (std::abs(a(0) - y_true) < tolerance) ++val_correct;
                }

            }
        

			// Calculate accuracy (Normalize metrics):
            accuracy = correct_predictions / static_cast<double>(train_inputs.size());
            val_accuracy = val_correct / static_cast<double>(val_inputs.size());
            epoch_loss /= static_cast<double>(train_inputs.size());
            val_loss /= static_cast<double>(val_inputs.size());


			// Check for early training termination (falling below min_delta):
            if (abs(best_val_loss - val_loss) > min_delta) {
                best_val_loss = val_loss;
                patience_counter = 0;
            }
            else {
                patience_counter++;
                if (patience_counter >= patience_limit) {
                    std::cout << "Early stopping after epoch " << epoch
                        << " (no significant improvement in " << patience_limit << " epochs)\n";
                    break;
                }
            }

            // End time measurement for the epoch:
            auto end_time = chrono::high_resolution_clock::now(); // Zeit stoppen
            chrono::duration<double> epoch_duration = end_time - start_time;

            // Always write to the log:
            log_epoch(log, epoch + 1, epoch_loss, accuracy, val_loss, val_accuracy, epoch_duration.count(), config.learning_rate);


            // Show only every 10 epochs:
            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << epoch + 1
                    << " - loss: " << epoch_loss
                    << " - accuracy: " << (accuracy * 100) << "%"
                    << " - val loss: " << val_loss
                    << " - val accuracy: " << (val_accuracy * 100) << "%"
                    << " - duration: " << epoch_duration.count() << "s\n\n"
                    << std::flush;
            }

            
		} // End of epoch training loop

        // ---------------  End of training ------------------------------

        // Zeitmessung für das gesamte Training beenden:
        auto training_end_time = chrono::high_resolution_clock::now(); // Zeit stoppen
        chrono::duration<double> training_duration = training_end_time - training_start_time;

        log.close(); // Log-Datei schließen

        // NN-Gewichte und Bias speichern:
        // Mögliche Alternative: save_all_weights_biases_csv(weights, biases, "run42");
        save_all_weights_biases_csv(weights, biases);

        // Info-Ausgabe über den letzten Status von ...
        std::cout << "Used / last (if dynamic) learning rate: " << config.learning_rate << endl;
		std::cout << "Training time: " << training_duration.count() << " seconds" << endl;


    }
    else if (config.modus == 2) {

        // ------------------------------------ Test-Modus ------------------------------

		// Modus 2: Testen eines trainierten Netzes
		cout << "Mode 2: Testing a trained network\n";

        // Netz-Konfiguration aus JSON laden:
        // Argumente zum Test anpassen:
        string testfile = config.dataset_file;       // z. B. mnist_test.csv
        string weight_prefix = "weights";            // z. B. run42          TODO: Unterverzeichnis für die Gewichte fest verdrahtet, muss überarbeitet werden!

		// Rest der Konfiguration aus der JSON-Datei laden:
        if (!load_config_from_json(config, "nn_parameter.json")) {
            std::cerr << "Error loading parameter file!\n";
            return -1;
        }

        // Verlustfunktion erstellen:
        unique_ptr<LossFunction> loss_fn = createLossFunction(config.loss_type);

        // Gewichte & Biases laden:
        vector<Eigen::MatrixXd> weights;
        vector<Eigen::VectorXd> biases;
        if (!load_all_weights_biases_csv(weights, biases, weight_prefix)) {
            cerr << "Error loading weights/biases!\n";
            return -1;
        }


        // Testdaten laden:
        DatasetInfo testdata = load_dataset_info(testfile, -1);
        std::vector<Eigen::VectorXd> X_test;
        std::vector<Eigen::VectorXd> y_test_enc;
        std::vector<int> y_test_int;
        std::vector<float> y_test_float;

        prepare_test_data(testdata, X_test, y_test_enc, y_test_int, y_test_float);

		// Testen des Netze

		// Validierungs- und Testgenauigkeit berechnen:
        std::vector<double> true_values;
        std::vector<double> predicted_values;

        for (size_t i = 0; i < X_test.size(); ++i) {
            Eigen::VectorXd prediction = predict(X_test[i], weights, biases, config.activations);

            if (testdata.is_classification) {
                int y_true = y_test_int[i];
                int y_pred = argmax(prediction);
                true_values.push_back(y_true);
                predicted_values.push_back(y_pred);
                if (y_pred == y_true)
                    ++val_correct;
            }
            else {
                double y_true = y_test_float[i];
                double y_pred = prediction(0);
                true_values.push_back(y_true);
                predicted_values.push_back(y_pred);
                if (std::abs(y_pred - y_true) < tolerance)
                    ++val_correct;
            }

            val_loss += loss_fn->compute(prediction, y_test_enc[i]);
        }

		// Ergebnisse ausgeben:
        save_predictions_csv("test_results.csv", true_values, predicted_values);

		// Ready Message:
		cout << "\nTest with " << X_test.size() << " Test samples completed. Results saved in 'test_results.csv'.\n";


		


    }
    else {

        // ----------------- Beenden -----------------
        // Modus 3: Programm beenden
        std::cout << "The program is terminated. Thank you for using EIGENnet!\n";
        std::cout << "For further information, please visit:\n";
        std::cout << "  YouTube: https://www.youtube.com/@r-statistik\n";
        std::cout << "  GitHub: https://github.com/SuprenumDE/Dense_NN";
		std::cout << "Program finished." << endl; 

    }

    return 0;

}
