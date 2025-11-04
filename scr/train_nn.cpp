// train_nn.cpp /Dense_NN
// Neuronales Netz vom Typ Dense auf Basis von "Eigen"
// 
// Entwickler: Guenter Faes, eigennet@faes.de
// GitHub: https://github.com/SuprenumDE/Dense_NN
// Lizenz: MIT (also for all self-developed included h and cpp files)
// Version 0.1.10, 25.09.2025
// Eigen-Version: 3.4.0
// C-Version: ISO-Standard C++20
// --------------------------------------

#include "Eigen/Eigen"
#include "cxxopts.hpp"              // Argumentparser
#include "weights_io.h"             // Gewichte und Bias einlesen/speichern
#include "activation_functions.h"   // Aktivierungsfunktionen
#include "loss_functions.h"         // Verlustfunktionen
#include "weights_ini.h"            // Gewichtsinitalisierung
#include "learning_rate_utils.h"    // Lernraten-Hilfsfunktionen
#include "dataset_utils.h"          // Datenhandling-Funktionen
#include "logging_utils.h"          // Logging-Funktionen
#include "prepare_data.h"           // Daten vorbereiten für Training/Validierung
#include "config.h"                 // Konfiguration des neuronalen Netzes        
#include "config_utils.h"           // Konfigurationsmanagement
#include "diagnostics_utils.h"      // Diagnosefunktionen für das Training
#include "metrics_utils.h"          // Metriken für Klassifikation/Regression (Hilfsfunktionen)
#include "resource.h"               // Icon und & 


#include <iostream>
#include <fstream>
#include <sstream>                  // Stringstream für CSV-Zeilen
#include <vector>
#include <cmath>                    // Mathematische Funktionen
#include <random>
#include <iomanip>                  // Formatierte Ausgabe
#include <chrono>                   // Zeitmessung
#include <stdexcept>                // Ausnahmebehandlung
#include <set>                      // Set für eindeutige Labels


using namespace std;

// Argumente parsen und in Config-Objekt speichern:
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

            // Aktivierungsfunktionen parsen
            std::stringstream ss_act(result["activations"].as<std::string>());
            std::string act;
            while (std::getline(ss_act, act, ',')) {
                config.activations.push_back(trim(act));
            }
            if (config.activations.size() != config.architecture.size() - 1) {
                std::cerr << "Error: The number of activations must be equal to the number of layer transitions.\n";
                return std::nullopt;
            }

            // Weitere Parameter
            config.weights_file = result["weights"].as<std::string>();
            config.epochs = result["epochs"].as<int>();
            config.n_TrainingsSample = result["samples"].as<int>();
            config.batch_size = result["batch"].as<int>();
            config.val_split = result["val"].as<float>();
            config.learning_rate = result["learning_rate"].as<double>();
            config.lr_mode = result["lr_mode"].as<std::string>();
            config.min_delta = result["min_delta"].as<double>();

            // Verlustfunktion parsen
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

            // Initialisierungsmethode parsen
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

            // Gültigkeitsprüfungen
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

EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.10, 25.09.2025, License: MIT
)";
    std::cout << "\nUsage:\n";
    std::cout << "  ./Dense_NN [Optionen]\n\n";
    std::cout << "Available options:\n";
    std::cout << options.help({ "" }) << "\n";
    std::cout << "Example:\n";
    std::cout << "  ./Dense_NN --architecture=784,128,64,10 --activations=relu,relu,softmax --epochs=100 --lr=0.01 --dataset=mnist.csv\n\n";
    std::cout << "Further information:\n";
    std::cout << "  Guenter Faes, Mail: eigennet@faes.de\n";
    std::cout << "  YouTube: https://www.youtube.com/@r-statistik\n";
    std::cout << "  GitHub:  https://github.com/SuprenumDE/EigenNET\n";
}

// ------------------------------- Ende Hilfsfunktionen -----------------





int main(int argc, char* argv[]) {

    // ----------------- Initialisierung -----------------
	vector<size_t> architecture;                    // Layergrößen, z. B. 784,128,64,10
	vector<std::string> activations;                // Aktivierungsfunktionen, z. B. relu,relu,tanh,softmax
    int modus = 1;                                  // Verabeitungsmodus : 1=Trainieren, 2=Testen, 3=Beenden
    int epochs = 0;                                 // Trainingsepoche, 0 = Default
    int n_TrainingsSample = 1000;                   // Um nur ein paar Trainingsdaten zu laden
    size_t batch_size = 64;                         // Minibatch-Größe, Standardwert
    float val_split = 0.2f;                         // Anteil der Daten für Validierung (z. B. 0.2)
    double learning_rate = 0.01;                    // Lernrate, Standardwert
    double dynamische_Lr = 0;                       // Nimmt die aktuelle dynamische Lernrate auf
    string lr_mode = {};                            // Dynamische Lernrate, z.B. "decay" oder "step"
    InitType W_init_Methode = InitType::HE;         // Initialisierungsmethode für die Gewichte, weights_ini.h
    LossType loss_type = LossType::CROSS_ENTROPY;   // Verlustfunktion, z.B. "CROSS_ENTROPY", "MAE" oder "MSE"

    // Kontrollvariablen
    double correct_predictions = 0;
    double accuracy = 0.0;
    double epoch_loss = 0.0;
    double val_loss = 0.0;
    double val_correct = 0;
    double val_accuracy = 0.0;
	double tolerance = 0.01;                        // Toleranz für die Genauigkeit, Vorbelegung 0.01

    //Variablen für frühes Stoppen des Trainings:
	double best_val_loss = 0.0;                 //  Beste Validierungs-Verlust
    int patience_counter = 0;                   // Zähler für Geduld
    const int patience_limit = 5;               // z. B. 5 Epochen Geduld
    double min_delta = 1e-4;                    // minimale Verbesserung

    // Dateihandling:
    string weights_file = "weights.txt";       // Dateiname für Gewichte und Bias
    string dataset_file = "mnist.csv";         // Standard-MNIST-Datensatz

	// Datenstrukturen Dataset-Utils
	DatasetInfo dataset; 				       // Struktur für Datensatz-Informationen

	// Neuronale Netz-Parameter: 
	vector<Eigen::MatrixXd> weights;           // Gewichte der Schichten
	vector<Eigen::VectorXd> biases;            // Biases der Schichten
	vector<Eigen::VectorXd> a_values;          // Aktivierungswerte der Schichten
	vector<Eigen::VectorXd> z_values;          // Z-Werte der Schichten (Voraktivierung)
	long long total_parameters = 0;            // Gesamtanzahl der Parameter im Netz

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
			("p,print_config", "Outputs the current configuration")
            ("h,help", "Displays help");
        

	//  Auswertung Argument-Optionen:
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
        Config config = *config_opt;  // Konfiguration aus den Argumenten

        if (result.count("print_config")) {
            print_config(config, std::cout);
        }
       

    if (config.modus == 1) {

		cout << "EIGENnet - Neural networks in C++ with Eigen,   Version 0.1.10, 25.09.2025\n";
        cout << "Training mode activated.\n\n";
        
        // ---------- Trainings-Modus ---------------------

         // Konfiguration in die real genutzten Variablen übertragen:
        dataset_file = config.dataset_file;
        architecture = config.architecture;
        activations = config.activations;
        weights_file = config.weights_file;
        epochs = config.epochs;
        n_TrainingsSample = config.n_TrainingsSample;
        batch_size = config.batch_size;
        val_split = config.val_split;
        learning_rate = config.learning_rate;
        lr_mode = config.lr_mode;
        W_init_Methode = config.W_init_Methode;
        min_delta = config.min_delta;
        loss_type = config.loss_type;

        // Daten laden und Dimensionen für Input- und Output-Layer automatisch bestimmen:
        DatasetInfo dataset = load_dataset_info(dataset_file, n_TrainingsSample);
        report_dataset_stats(dataset, val_split); // Ausgabe der Statistik des Datensatzes

        // Aktivierungsfunktionen validieren:
        for (const auto& act_name : activations) {
            try {
                Activation::get(act_name); // Testweise abrufen
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
			architecture[0] = dataset.input_dim;            // Input-Layer-Größe anpassen
            std::cout << "Warning: Input layer size adjusted to match dataset input dimension: " << dataset.input_dim << "\n\n";
        }
		
        // Initialization of weights:

        // Check whether iterative training, load weights from previous training:

        std::vector<Eigen::MatrixXd> pretrained_weights;
        std::vector<Eigen::VectorXd> pretrained_biases;

        if (W_init_Methode == InitType::ITERATIVE) {

			std::cout << "Iterative training selected. Attempting to load pre-trained weights...\n\n";

            // Loading the net weights from the previous training session:
            string weight_prefix = "weights";            // z. B. run42          TODO: Unterverzeichnis für die Gewichte fest verdrahtet, muss überarbeitet werden!
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
                Eigen::VectorXd b(output_dim);                     // Bias-Vektor für die Schichten
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
        cout << flush;

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
        if(dataset.is_classification) {
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
       


        // Start training:

        // Zeitmessung für das gesamte Training:
        auto training_start_time = chrono::high_resolution_clock::now();

        // Logbuch anlegen:
		ofstream log("training_log.csv");
        if (!log.is_open()) {
            cerr << "Error: Log file could not be opened." << endl;
            return 1;
		}
		// Netzparameter als JSON-Datei speichern:
        save_config_as_json(config, "nn_parameter.json");

        
		// Logbuch-Header schreiben:
        write_log_header(log);

        // Trainingsschleife:
        for (int epoch = 0; epoch < epochs; ++epoch) {

            // Zeitmessung für die Epoche:
            auto start_time = chrono::high_resolution_clock::now();

            epoch_loss = 0.0;
            correct_predictions = 0;
            val_loss = 0.0;
            val_correct = 0;


            // Batch-Training: 
            for (size_t i = 0; i < train_inputs.size(); i += batch_size) {

                size_t end = std::min(i + batch_size, train_inputs.size());

				// Initialisierung der Aktivierungen und Zwischenergebnisse:
                vector<Eigen::MatrixXd> dw_acc(weights.size());
                vector<Eigen::VectorXd> db_acc(biases.size());
                for (size_t l = 0; l < weights.size(); ++l) {
                    dw_acc[l] = Eigen::MatrixXd::Zero(weights[l].rows(), weights[l].cols());
                    db_acc[l] = Eigen::VectorXd::Zero(biases[l].size());
                }


                // Schleife über das aktuelle Minibatch:
				for (size_t j = i; j < end; ++j) { 

                    // Vorwärts:
                    // Temporäre Variablen für die Batch-Verarbeitung löschen (Speicher freigeben):
                    a_values.clear();
                    z_values.clear();

                    Eigen::VectorXd a = train_inputs[j];  // Input
                    a_values.push_back(a);     

                    for (size_t l = 0; l < weights.size(); ++l) {

                        Eigen::VectorXd z = weights[l] * a + biases[l];
                        z_values.push_back(z);
                        auto act = Activation::get(activations[l]);
                        a = act->activate(z);
                        a_values.push_back(a);

                    }


                    // Rückwärts:
					vector<Eigen::VectorXd> deltas(weights.size()); // Deltas für Backpropagation

                    // Delta-Berechnung:
                    Eigen::VectorXd prediction = a_values.back();
                    Eigen::VectorXd target = train_labels_enc[j];                            // Bestimmen des Zielwert
					Eigen::VectorXd dz = loss_fn->gradient(prediction, target);              // Gradient des Verlusts
                    deltas.back() = dz;

                    // Diagnose: Gradient der letzten Schicht
                    // std::cout << "[Diagnose] Delta Output Layer-Norm: " << dz.norm() << "\n";

					// Backpropagation durch die Schichten:
                    for (int l = static_cast<int>(weights.size()) - 2; l >= 0; --l) {
						Eigen::VectorXd da = weights[l + 1].transpose() * deltas[l + 1];    
                        auto act = Activation::get(activations[l]);
						Eigen::VectorXd dz = da.array() * act->derivative(z_values[l]).array(); // Ableitung der Aktivierungsfunktion
                        deltas[l] = dz;

					} // Ende Backpropagation durch die Schichten

                    // Diagnose: Gradientennormen aller Schichten
                    // show_gradient_norms(deltas);

 
					// Gradientenakkumulation:
                    for (size_t l = 0; l < weights.size(); ++l) {
						dw_acc[l] += deltas[l] * a_values[l].transpose(); // Gradienten für Gewichte
						db_acc[l] += deltas[l];                           // Gradienten für Biases
                    }


                    // Verlust & Accuracy
                    epoch_loss += loss_fn->compute(prediction, target);


					// Korrektklassifizierung zählen:
                    if (dataset.is_classification) {
                        // Klassifikation: index der größten Wahrscheinlichkeit
                        int y_true = train_labels_int[j];
                        if (argmax(prediction) == y_true)
                            ++correct_predictions;
                    }
                    else {
                        // Regression: Abweichung gegen das 1×1-Label in train_labels_enc/train_labels_float
						// Hier wird angenommen, dass train_labels_float ein 1D-Vektor ist.
                        double y_true = train_labels_float[j];
                        if (std::abs(prediction(0) - y_true) < tolerance)
                            ++correct_predictions;
                    }


				} // Ende Trainingsdaten-Schleife


                // Lernrate anpassen
                double adjusted_lr = adjust_learning_rate(lr_mode, learning_rate, epoch);
				dynamische_Lr = adjusted_lr; // Aktuelle dynamische Lernrate speichern  

                // Update mit Mittelwert über Batch
                for (size_t l = 0; l < weights.size(); ++l) {
                    weights[l] -= adjusted_lr * (dw_acc[l] / batch_size);
                    biases[l] -= adjusted_lr * (db_acc[l] / batch_size);
                }


			} // Ende der Netzschleife

			// Validierungsverlust und -genauigkeit berechnen:
            for (size_t i = 0; i < val_inputs.size(); ++i) {

                Eigen::VectorXd a = val_inputs[i];

                for (size_t l = 0; l < weights.size(); ++l) {
                    Eigen::VectorXd z = weights[l] * a + biases[l];
                    auto act = Activation::get(activations[l]);
                    a = act->activate(z);
                }

                Eigen::VectorXd target = val_labels_enc[i];

				// Validierungsverlust berechnen:
                val_loss += loss_fn->compute(a, target);

				// Korrektklassifizierung zählen:
                if (dataset.is_classification) {
                    int y_true = val_labels_int[i];
                    if (argmax(a) == y_true) ++val_correct;
                }
                else {
                    double y_true = train_labels_float[i];
                    if (std::abs(a(0) - y_true) < tolerance) ++val_correct;
                }

            }

			// Genauigkeit berechnen:
            accuracy = correct_predictions / static_cast<double>(train_inputs.size());
            val_accuracy = val_correct / static_cast<double>(val_inputs.size());
            epoch_loss /= static_cast<double>(train_inputs.size());
            val_loss /= static_cast<double>(val_inputs.size());


			// Prüfung zum frühen Trainingsstopp (Unterschreiten von min_delta):
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

            // Zeitmessung für die Epoche beenden:
            auto end_time = chrono::high_resolution_clock::now(); // Zeit stoppen
            chrono::duration<double> epoch_duration = end_time - start_time;

            // Immer ins Log schreiben:
            log_epoch(log, epoch + 1, epoch_loss, accuracy, val_loss, val_accuracy, epoch_duration.count(), dynamische_Lr);


            // Nur alle 10 Epochen anzeigen:
            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << epoch + 1
                    << " - loss: " << epoch_loss
                    << " - accuracy: " << (accuracy * 100) << "%"
                    << " - val loss: " << val_loss
                    << " - val accuracy: " << (val_accuracy * 100) << "%"
                    << " - duration: " << epoch_duration.count() << "s\n\n"
                    << std::flush;
            }

            
		} // Ende der Trainingsschleife

        // Zeitmessung für das gesamte Training beenden:
        auto training_end_time = chrono::high_resolution_clock::now(); // Zeit stoppen
        chrono::duration<double> training_duration = training_end_time - training_start_time;

        log.close(); // Log-Datei schließen

        // NN-Gewichte und Bias speichern:
        // Mögliche Alternative: save_all_weights_biases_csv(weights, biases, "run42");
        save_all_weights_biases_csv(weights, biases);

        // Info-Ausgabe über den letzten Status von ...
        cout << "Used / last (if dynamic) learning rate: " << dynamische_Lr << endl;
		cout << "Training time: " << training_duration.count() << " seconds" << endl;


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
        std::cout << "  GitHub: ";
		std::cout << "Program finished." << endl; 

    }

    return 0;

}
