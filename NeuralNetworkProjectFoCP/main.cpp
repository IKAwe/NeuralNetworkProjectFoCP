#include <vector>
#include <iostream>
#include "file_data_operations.h"
#include "neural_network.h"

struct Configuration {
    std::string train_data_path = "iris.csv";
    std::string test_data_path = "";
    std::string model_save_path = "weights.txt";
    int epochs = 10;
    double learning_rate = 0.5;
    int batch_size = 8;
};
Configuration parseArguments(int argc, char* argv[]) {
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
    Configuration config = parseArguments(argc, argv);


    auto data = parseCSV(config.train_data_path);
    std::vector<int> input_data;

    /*for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }*/


	NeuralNetwork nn;
    nn.initialize_weights(2, 3, 3, 2);
    nn.visualize_weights();
    nn.save_weights_to_file(config.model_save_path);
    nn.load_weights_from_file(config.model_save_path);
    nn.visualize_weights();

	//nn.load_weights_from_file("weights.txt");
	//std::cout << nn.feedforward(data[1]) << std::endl;
	//nn.save_weights_to_file("weights.txt");
    return 0;
}
