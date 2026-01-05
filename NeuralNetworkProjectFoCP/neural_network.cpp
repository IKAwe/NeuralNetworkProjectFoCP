#include "neural_network.h"
#include "file_data_operations.h"
#include "math_functions.h"
#include <random>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "display.h"

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::initialize_weights_and_biases(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size) {
    weights.clear();
    weights.resize(hidden_layers_number + 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    int prev_size = input_size;

    // Hidden layers
    for (int i = 0; i < hidden_layers_number; ++i) {
        weights[i].resize(neurons_per_hidden_layer);
        for (int j = 0; j < neurons_per_hidden_layer; ++j) {
            weights[i][j].resize(prev_size);
            for (int k = 0; k < prev_size; ++k) {
                weights[i][j][k] = distribution(generator);
            }
        }
        prev_size = neurons_per_hidden_layer;
    }

    // Output layer
    weights[hidden_layers_number].resize(output_size);
    for (int j = 0; j < output_size; ++j) {
        weights[hidden_layers_number][j].resize(prev_size);
        for (int k = 0; k < prev_size; ++k) {
            weights[hidden_layers_number][j][k] = distribution(generator);
        }
    }
	// Biases initialization (optional)
    biases.clear();
    biases.resize(hidden_layers_number + 1);
    for (int i = 0; i < hidden_layers_number; ++i) {
        biases[i].resize(neurons_per_hidden_layer);
        for (int j = 0; j < neurons_per_hidden_layer; ++j) {
            biases[i][j] = distribution(generator);
        }
    }
    biases[hidden_layers_number].resize(output_size);
    for (int j = 0; j < output_size; ++j) {
        biases[hidden_layers_number][j] = distribution(generator);
	}

	is_model_valid = true;

	// Initialize activations for each layer except input layer
	activations.clear();
    for (int i = 0; i < hidden_layers_number+1; ++i) {
        activations.push_back("relu"); // Example activation 
	}
}

void NeuralNetwork::visualize_model() const {
	std::cout << "\n";
    print_header1("Neural Network Weight Structure:");
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        std::cout << "\nLayer " << layer + 1 << " (" << weights[layer].size() << " neurons):\n";
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            std::cout << "  Neuron " << neuron + 1 << " weights: ";
            for (size_t w = 0; w < weights[layer][neuron].size(); ++w) {
                std::cout << std::fixed << std::showpos<< std::setprecision(2) << "\t" << weights[layer][neuron][w] << " ";
            }
            std::cout << "\n";
        }
    }
	// visualize biases
    print_header1("Neural Network Biases Structure:");
    for (size_t layer = 0; layer < biases.size(); ++layer) {
        std::cout << "\nLayer " << layer + 1 << " biases: ";
        for (size_t neuron = 0; neuron < biases[layer].size(); ++neuron) {
            std::cout << std::fixed << std::setprecision(2) << "\t" << biases[layer][neuron] << " ";
        }
        std::cout << "\n";
	}
	std::cout <<std::noshowpos;
}

/**
 * @brief Get the output of the neural network for a given input.
 * @param input The input vector.
 * @return The output vector after passing through the network.
 */
std::vector<std::vector<float>> NeuralNetwork::feedforward(const std::vector<std::vector<float>>& input) const {
    if (!is_model_valid) {
        std::cout << "Model structure is invalid. Cannot perform feedforward.\n";
        return { {} };
	}
	std::vector<std::vector<float>> output;
    output.reserve(input.size());
    for (const auto& record : input) {
        std::vector <float> prev_layer_output = record;
        for (int layer_nb = 0; layer_nb < weights.size(); ++layer_nb) {
            const auto& layer = weights[layer_nb];
            std::vector<float> curr_layer_output(layer.size());
            for (int neuron_nb = 0; neuron_nb < layer.size(); ++neuron_nb) {
                //print_vector(curr_layer_output);
                float sum = biases[layer_nb][neuron_nb];

                for (int weight_nb = 0; weight_nb < layer[neuron_nb].size(); ++weight_nb) {
                    sum += layer[neuron_nb][weight_nb] * prev_layer_output[weight_nb];
                }
                curr_layer_output[neuron_nb] = sum;
            }
            // Apply activation function
            curr_layer_output = activation_map.at(activations[layer_nb]).func(curr_layer_output); //activation_map is const
            print_vector(curr_layer_output);
            prev_layer_output = curr_layer_output;
        }
		output.push_back(prev_layer_output);
    }
	return output;
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs,
                          const std::vector<std::vector<float>>& targets,
                          int epochs,
                          float learning_rate,
                          const std::string& loss_function_name) {
    // Training logic to be implemented
	std::vector<float> loss;
	std::vector<float> output;
    for (int epoch = 0; epoch < epochs; ++epoch) {
		output = feedforward(inputs[epoch % inputs.size()]);
        for(int layer = weights.size() - 1; layer >= 0; --layer) {
            // Backpropagation logic to be implemented
		}
    }
}

/**
 * @brief  Load weights from a file. (each line corresponds to a neuron's weights, each weight is separated by a comma, each layer is separated by an empty line)
 * @param filename The name of the file to load the weights from.
 */

void NeuralNetwork::save_model_to_file(const std::string& filename) const {
	std::cout << "\nSaving model to file base name - " << filename << " ...\n";
    save_vector_to_file(filename + "_weights.txt", weights);
	save_vector_to_file(filename + "_biases.txt", biases);
    save_vector_to_file(filename + "_activations.txt", activations);
	std::cout << "\nModel was successfully saved to file base name: " << filename << "\n";
}



/**
 * @brief  Load weights from a file. (each line corresponds to a neuron's weights, each weight is separated by a comma)
 * @param filename The name of the file to load the weights from.
 */

void NeuralNetwork::load_model_from_file(const std::string& filename) {
	std::cout << "\nTrying to load model from file base name: " << filename << " ...\n";
    load_vector_from_file(filename + "_weights.txt", weights);
    load_vector_from_file(filename + "_biases.txt", biases);
	load_vector_from_file(filename + "_activations.txt", activations);
    std::cout << "Model was successfully read from file base name: " << filename << "\n";

	validate_model_structure();
}

/**
 * @brief Validates weights, biases, and activations sizes. If not valid, sets is_model_valid to false.
 */
void NeuralNetwork::validate_model_structure() {
	auto input_size = weights[0][0].size();
	auto hidden_layers_number = weights.size() - 1;
	auto neurons_per_hidden_layer = weights[0].size();
	auto output_size = weights[hidden_layers_number].size();

    /*if (weights.size() != (size_t)(hidden_layers_number + 1)) {
		is_model_valid = false;
		std::cout << "Invalid number of layers in the model\n";
		return;
    }*/
    // Validate hidden layers

	std::cout << "\nValidating model structure...\n";
    for (int i = 0; i < hidden_layers_number; ++i) {
        if (weights[i].size() != (size_t)(neurons_per_hidden_layer)) {
            is_model_valid = false;
            std::cout << "Invalid number of neurons in hidden layer " << (i + 1) << "\n";
        }
        for (const auto& neuron_weights : weights[i]) {
            if (neuron_weights.size() != (size_t)(i == 0 ? input_size : neurons_per_hidden_layer)) {
				std::cout << "Invalid number of weights for a neuron in hidden layer " << (i + 1) << "\n";
				is_model_valid = false;
				return;
            }
        }
    }
    // Validate output layer
    if (weights[hidden_layers_number].size() != (size_t)(output_size)) {
		std::cout << "Invalid number of neurons in output layer\n";
		is_model_valid = false;
        return;
    }
    for (const auto& neuron_weights : weights[hidden_layers_number]) {
        if (neuron_weights.size() != (size_t)(neurons_per_hidden_layer)) {
			std::cout << "Invalid number of weights for a neuron in output layer\n";
            is_model_valid = false;
			return;
        }
    }


    //Validate biases
	if (biases.size() != weights.size()) {
        is_model_valid = false;
        std::cout << "Invalid number of bias layers\n";
		return;
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        if (biases[i].size() != weights[i].size()) {
            is_model_valid = false;
            std::cout << "Invalid number of biases in layer " << (i + 1) << "\n";
            return;
        }
    }

	//validate activations
    if (activations.size() != weights.size()-1) {
        is_model_valid = false;
        std::cout << "Invalid number of activation functions\n";
        return;
	}
    
    for (const auto& act : activations) {
		if(activation_map.find(act) ==  activation_map.end()) {
            is_model_valid = false;
            std::cout << "Unknown activation function: " << act << "\n";
            return;
        }
	}
    if (activations.size() != weights.size() - 1) {
        is_model_valid = false;
        std::cout << "Mismatch between number of activation functions and layers\n";
        return;
	}

	std::cout << "Current model structure is valid \n";
}


