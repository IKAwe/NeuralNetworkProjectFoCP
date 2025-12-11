#include <vector>
#include <iostream>
#include "file_data_operations.h"
#include "neural_network.h"
#include "data_preprocessor.h"
#include "display.h"

struct Configuration {
    std::string train_data_path = "Exam_Score_Prediction.csv";
    std::string test_data_path = "";
    std::string model_save_path = "model";
    int epochs = 10;
    float learning_rate = 0.5;
    int batch_size = 8;
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
        else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        }
    }

    return config;
}

int main(int argc, char* argv[]) {
    Configuration config = parse_arguments(argc, argv);


	std::unordered_map<std::string, std::vector<std::vector<float>>> all_data_splits;
	//Dataset loading
	print_header2("DATA LOADING AND PREPROCESSING");
    std::vector<std::string> column_names;
    std::vector<std::vector<std::string>> dataset = parseCSV(config.train_data_path);

    DataPreprocessor dp;

    // Test for extracting target column for neural network
	std::string target_column = "age";

    dp.fit(dataset);
    dp.print_state();

    auto transformed_dataset = dp.transform_and_extract(dataset, target_column);
	std::cout << "Transformed dataset (features):" << std::endl;
    for (int i = 0; i < 3; ++i) {
		print_vector(transformed_dataset.first[i]);
	}
    std::cout << "Transformed dataset (targets):" << std::endl;
    for (int i = 0; i < 3; ++i) {
		print_vector(transformed_dataset.second[i]);
	}
	print_header2("Neural Network Model Initialization and Loading");
	int input_size = transformed_dataset.first[0].size();
	int output_size = transformed_dataset.second[0].size();


    /*for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }*/


	NeuralNetwork nn;
    nn.initialize_weights_and_biases(input_size, 3, 4, output_size);
    nn.visualize_model();
    /*nn.save_model_to_file(config.model_save_path);
    std::cout << std::endl;
    nn.load_model_from_file(config.model_save_path);
    nn.visualize_model();*/

	//std::cout << nn.feedforward(data[1]) << std::endl;
    return 0;
}
