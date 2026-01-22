#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include "file_data_operations.h"
#include "neural_network.h"
#include "data_preprocessor.h"
#include "display.h"

struct Configuration {
    std::string train_data_path = "iris.csv";
	std::string target_column = "species";
    std::string test_data_path = "";
    std::string model_save_path = "model";
	std::string model_load_path = "model";
    int epochs = 140;
    float learning_rate = 0.3;
    float test_fraction = 0.2;
};

Configuration parse_arguments(int argc, char* argv[]) {
    Configuration config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--train" && i + 1 < argc) {
            config.train_data_path = argv[++i];
        }
        else if (arg == "--test" && i + 1 < argc) {
            config.test_data_path = argv[++i];
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stod(argv[++i]);
        }
        else if (arg == "--test_fraction" && i + 1 < argc) {
            config.test_fraction = std::stof(argv[++i]);
        }
        else if (arg == "--target_column" && i + 1 < argc) {
            config.target_column = argv[++i];
        }
        else if (arg == "--save_model" && i + 1 < argc) {
            config.model_save_path = argv[++i];
        }
        else if (arg == "--load_model" && i + 1 < argc) {
            config.model_load_path = argv[++i];
		}
    }

    return config;
}

int main(int argc, char* argv[]) {
	/*NeuralNetwork nn;
	nn.initialize_weights_and_biases(2, 1, 2, 1, "He", {"relu", "sigmoid"});
	std::vector<std::vector<float>> inputs = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
	std::vector<std::vector<float>> targets = { {0.0f}, {1.0f}, {1.0f}, {0.0f} };
	nn.train(inputs, targets, inputs, targets, 10000, 0.1f, "MSE");
	print_feedforward_output(inputs, targets, nn.feedforward(inputs));
	nn.save_model_to_file("model_for_testing");
	nn.save_model_to_file("xor_model");*/
    Configuration config = parse_arguments(argc, argv);

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
    nn.initialize_weights_and_biases(input_size, 4, 5, output_size, "Xavier", {"sigmoid","sigmoid", "tanh", "sigmoid", "sigmoid"});
	nn.train(train_inputs,
		    train_targets,
		    test_inputs,
		    test_targets,
             config.epochs,
             config.learning_rate,
             "MSE");


    //See example
	print_feedforward_output(test_inputs,test_targets,nn.feedforward(test_inputs));

	dp.interpret_extracted_column(nn.feedforward(test_inputs)[0]);

	//nn.save_model_to_file(config.model_save_path);
    return 0;
}
