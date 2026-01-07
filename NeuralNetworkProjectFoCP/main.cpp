#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include "file_data_operations.h"
#include "neural_network.h"
#include "data_preprocessor.h"
#include "display.h"

struct Configuration {
    std::string train_data_path = "Exam_Score_Prediction.csv";
    std::string test_data_path = "";
    std::string model_save_path = "model";
    int epochs = 30;
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

	//Dataset loading
    std::vector<std::string> column_names;
    std::vector<std::vector<std::string>> dataset = parseCSV(config.train_data_path);

    //Shuffle dataset
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(dataset)+1, std::end(dataset), rng);

    DataPreprocessor dp;

    // Test for extracting target column for neural network
	std::string target_column = "exam_score";

    dp.fit(dataset);
    dp.print_state();

    auto transformed_dataset = dp.transform_and_extract(dataset, target_column);
    if (transformed_dataset.first.empty() || transformed_dataset.second.empty()) {
        std::cerr << "Error: Transformed dataset is empty.\n";
        return 0;
    }
	std::cout << "Transformed dataset (features):" << std::endl;
    for (int i = 0; i < std::min(3, (int)transformed_dataset.first.size()); ++i) {
		print_vector(transformed_dataset.first[i]);
	}
    std::cout << "Transformed dataset (targets):" << std::endl;
    for (int i = 0; i < std::min(3, (int)transformed_dataset.second.size()); ++i) {
		print_vector(transformed_dataset.second[i]);
	}
    
	int input_size = transformed_dataset.first[0].size();
	int output_size = transformed_dataset.second[0].size();


	//TRAINING
	NeuralNetwork nn;
    nn.initialize_weights_and_biases(input_size, 3, 4, output_size);
    nn.train(transformed_dataset.first,
             transformed_dataset.second,
             config.epochs,
             config.learning_rate,
             "MSE",
		0.2f);

    //nn.visualize_model();
	//std::cout << "Feedforward result for the first sample:" << std::endl;

    /*nn.save_model_to_file(config.model_save_path);
    std::cout << std::endl;
    nn.load_model_from_file(config.model_save_path);
    nn.visualize_model();*/

	//std::cout << nn.feedforward(data[1]) << std::endl;
	//TESTING
    std::vector<std::vector<float>> test_sample = { transformed_dataset.first[1] };
	std::cout << "\nFeedforward sample input: ";
    print_vector(test_sample[0]);
	std::cout << "Feedforward sample prediction: ";
	print_vector(nn.feedforward(test_sample)[0]);
	std::cout <<"Correct output: ";
	print_vector(transformed_dataset.second[1]);
    return 0;
}
