#include <vector>
#include <iostream>
#include "file_data_operations.h"
#include "neural_network.h"
#include "data_preprocessor.h"

struct Configuration {
    std::string train_data_path = "iris.csv";
    std::string test_data_path = "";
    std::string model_save_path = "model";
    int epochs = 10;
    float learning_rate = 0.5;
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

    auto dataset = parseCSV(config.train_data_path);

    // Create preprocessor
    DataPreprocessor preprocessor;

    // First fit categories (so all rows expand to same length)
    preprocessor.fit(dataset);

    // Transform dataset into floats
    std::vector<std::vector<float>> processed = preprocessor.encode(dataset);

    // Print results
    std::cout << "Processed dataset:\n";
    for (const auto& row : processed) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }


    // Get information about output features
    std::cout << "Feature mapping:\n";
    for (int i = 0; i < preprocessor.get_output_size(); i++) {
        int input_col, cat_idx;
        std::string feature_name;
        preprocessor.get_feature_info(i, input_col, cat_idx, feature_name);
        std::cout << "Output " << i << ": " << feature_name
            << " (input col " << input_col << ", cat idx " << cat_idx << ")\n";
    }

    // Get range for specific columns
    std::cout << "\nColumn ranges:\n";
    for (int col = 0; col < preprocessor.get_input_columns(); col++) {
        int start, size;
        preprocessor.get_output_range(col, start, size);
        std::cout << preprocessor.get_column_name(col) << ": indices "
            << start << " to " << start + size - 1 << "\n";
    }


    //auto data = parseCSV(config.train_data_path);

    /*for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }*/


	/*NeuralNetwork nn;
    nn.initialize_weights_and_biases(3, 3, 4, 2);
    nn.visualize_model();
    nn.save_model_to_file(config.model_save_path);
    std::cout << std::endl;
    nn.load_model_from_file(config.model_save_path);
    nn.visualize_model();*/

	//std::cout << nn.feedforward(data[1]) << std::endl;
    return 0;
}
