#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include "file_data_operations.h"
#include "neural_network.h"
#include "data_preprocessor.h"
#include "display.h"
#include <sstream>


/**
 * @struct Configuration
 * @brief Holds all runtime settings for the neural network tool.
 * @details This structure is populated by the command-line parser and determines
 * the architecture, data paths, and training hyperparameters.
 */
struct Configuration {
    // Model Architecture
    std::string initialization_method = "Xavier";
    int hidden_layers = 1;
    int neurons_nb = 2;
    std::vector<std::string> activations = {};

    // Data 
    std::string train_data_path = "iris.csv";
    std::string target_column = "species";
    float test_fraction = 0.2f;

    // Hyperparameters
    int epochs = 140;
    float learning_rate = 0.3f;

    // File I/O
    std::string model_save_path = "";
    std::string model_load_path = "";
};

/**
 * @brief Displays the help manual and command-line usage instructions.
 */
void print_manual() {
    std::cout << "====================================================\n";
    std::cout << "   NEURAL NETWORK                                   \n";
    std::cout << "====================================================\n\n";

    std::cout << "USAGE:\n";
    std::cout << "  ./nn_tool [TRAINING OPTIONS] [MODEL OPTIONS] [ACTION OPTIONS]\n\n";

    std::cout << "MODEL OPTIONS (Initialize or Load):\n";
    std::cout << "  --initialization <name> Name of the initialization method (e.g. He, Xavier)\n";
    std::cout << "  --load_model <path>     Load weights from a file(if not specified either initialization or load_model model defaultly will be initialized\n";
    std::cout << "  --save_model <path>     Where to save the model after execution.\n\n";

    std::cout << "--hidden_layers <int>   Number of hidden layers (default: 1).\n";
    std::cout << "--neurons_nb <int>     Number of neurons in each hidden layer (default: 2).\n";
    std::cout << "--activations <list>   Comma-separated list of activation functions for each hidden layer and output layer.\n\n";


    std::cout << "TRAINING OPTIONS:\n";
    std::cout << "  --data_path <path>          Path to CSV file \n";
    std::cout << "  --target_column <name>  The column to predict \n";
    std::cout << "  --test_fraction <0-1>   Fraction of data for validation (default: 0.2)\n\n";


    std::cout << "HIPERPARAMETERS:\n";
    std::cout << "  --epochs <int>          Number of training iterations. Set to 0\n";
    std::cout << "                          to skip training and just run inference.\n";
    std::cout << "  --lr <float>            Learning rate (default: 0.3).\n\n";


    std::cout << "====================================================\n";
}

/**
 * @brief Parses command-line arguments into a Configuration object.
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return A populated Configuration struct with user-defined/default values.
 */
Configuration parse_arguments(int argc, char* argv[]) {
    Configuration config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        try {// Architecture
            if (arg == "--initialization" && i + 1 < argc) {
                config.initialization_method = argv[++i];
            }
            else if (arg == "--hidden_layers" && i + 1 < argc) {
                config.hidden_layers = std::stoi(argv[++i]);
            }
            else if (arg == "--neurons_nb" && i + 1 < argc) {
                config.neurons_nb = std::stoi(argv[++i]);
            }
            else if (arg == "--activations" && i + 1 < argc) {
                std::string line = argv[++i];
                std::stringstream ss(line);
                std::string func;
                config.activations.clear();
                while (std::getline(ss, func, ',')) {
                    config.activations.push_back(func);
                }
            }
            // Training Params
            else if (arg == "--data_path" && i + 1 < argc) {
                config.train_data_path = argv[++i];
            }
            else if (arg == "--target_column" && i + 1 < argc) {
                config.target_column = argv[++i];
            }
            else if (arg == "--test_fraction" && i + 1 < argc) {
                config.test_fraction = std::stof(argv[++i]);
            }
            else if (arg == "--epochs" && i + 1 < argc) {
                config.epochs = std::stoi(argv[++i]);
            }
            else if (arg == "--lr" && i + 1 < argc) {
                config.learning_rate = std::stof(argv[++i]);
            }
            // File I/O
            else if (arg == "--save_model" && i + 1 < argc) {
                config.model_save_path = argv[++i];
            }
            else if (arg == "--load_model" && i + 1 < argc) {
                config.model_load_path = argv[++i];
            }
            else if (arg == "--help" || arg == "-h") {
                config.train_data_path = ""; // Trigger manual
                return config;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: Invalid value for argument '" << arg << "'. Expected a number.\n";
            config.train_data_path = ""; // Trigger manual
            return config;
        }
    }
    return config;
}

int main(int argc, char* argv[]) {
    
    Configuration config = parse_arguments(argc, argv);
    if (config.train_data_path.empty()) {
        print_manual();
        return 0;
    }
    
	//===== DATA PREPROCESSING =====
    std::vector<std::string> column_names;
    std::vector<std::vector<std::string>> dataset = parseCSV(config.train_data_path);

    //Shuffle dataset
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(dataset)+1, std::end(dataset), rng);

    DataPreprocessor dp;

    // Test for extracting target column for neural network

    dp.fit(dataset);
    dp.print_state();

    auto transformed_dataset = dp.transform_and_extract(dataset, config.target_column);
    if (transformed_dataset.first.empty()) {
        std::cerr << "Error: No data available after preprocessing.\n";
        return 1;
    }
	//Divide to train and test sets
	int number_of_train_samples = transformed_dataset.first.size() * (1 - config.test_fraction);
    const std::vector<std::vector<float>>& train_inputs{ transformed_dataset.first.begin(), transformed_dataset.first.begin() + number_of_train_samples };
    const std::vector<std::vector<float>>& train_targets{ transformed_dataset.second.begin(), transformed_dataset.second.begin() + number_of_train_samples };

    const std::vector<std::vector<float>>& test_inputs{ transformed_dataset.first.begin() + number_of_train_samples, transformed_dataset.first.end() };
    const std::vector<std::vector<float>>& test_targets{ transformed_dataset.second.begin() + number_of_train_samples, transformed_dataset.second.end() };

	//========== TRAINING ==========
	NeuralNetwork nn;
    int input_size = transformed_dataset.first[0].size();
    int output_size = transformed_dataset.second[0].size();
    nn.initialize_weights_and_biases(input_size, config.hidden_layers, config.neurons_nb, output_size, config.initialization_method, config.activations);
	nn.train(train_inputs,
		    train_targets,
		    test_inputs,
		    test_targets,
             config.epochs,
             config.learning_rate,
             "BCE");


    //See example
	print_feedforward_output(test_inputs,test_targets,nn.feedforward(test_inputs));

	dp.interpret_extracted_column(nn.feedforward(test_inputs)[0]);

	if (!config.model_save_path.empty()) 
    {
        nn.save_model_to_file(config.model_save_path);
    }
    return 0;
}
