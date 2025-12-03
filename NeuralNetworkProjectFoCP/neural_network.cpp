#include "neural_network.h"
#include "file_data_operations.h"
#include "math_functions.h"
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

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
}

void NeuralNetwork::visualize_model() const {
	std::cout << "\n";
    std::cout << "Neural Network Weight Structure:\n";
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        std::cout << "\nLayer " << layer + 1 << " (" << weights[layer].size() << " neurons):\n";
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            std::cout << "  Neuron " << neuron + 1 << " weights: ";
            for (size_t w = 0; w < weights[layer][neuron].size(); ++w) {
                std::cout << std::fixed << std::setprecision(3) << weights[layer][neuron][w] << " ";
            }
            std::cout << "\n";
        }
    }
	std::cout << "\n";
	// visualize biases
    std::cout << "Neural Network Biases Structure:\n";
    for (size_t layer = 0; layer < biases.size(); ++layer) {
        std::cout << "\nLayer " << layer + 1 << " biases: ";
        for (size_t neuron = 0; neuron < biases[layer].size(); ++neuron) {
            std::cout << std::fixed << std::setprecision(3) << biases[layer][neuron] << " ";
        }
        std::cout << "\n";
	}
}

float NeuralNetwork::feedforward(const std::vector<float>& input) const {
    if (!is_model_valid) {
        std::cout << "Model structure is invalid. Cannot perform feedforward.\n";
        return 0.0f;
	}
    std::vector<float> activations = input;
    for (const auto& layer : weights) {
        std::vector<float> new_activations(layer.size(), 0.0f);
        for (size_t neuron = 0; neuron < layer.size(); ++neuron) {
            for (size_t w = 0; w < layer[neuron].size(); ++w) {
                new_activations[neuron] += layer[neuron][w] * activations[w];
            }
            // Apply activation function (e.g., sigmoid)
            //new_activations[neuron] = 1.0f / (1.0f + std::exp(-new_activations[neuron]));
        }
        activations = new_activations;
    }
    return activations.empty() ? 0.0f : activations[0];
}



/**
 * @brief  Load weights from a file. (each line corresponds to a neuron's weights, each weight is separated by a comma, each layer is separated by an empty line)
 * @param filename The name of the file to load the weights from.
 */

void NeuralNetwork::save_model_to_file(const std::string& filename) const {
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
    load_vector_from_file(filename + "_weights.txt", weights);
    load_vector_from_file(filename + "_biases.txt", biases);
	load_vector_from_file(filename + "_activations.txt", activations);
    std::cout << "\nModel was successfully read from file base name: " << filename << "\n";

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
		auto possible_activations = get_possible_activations();
        if (get_index(possible_activations, act) == -1) {
            is_model_valid = false;
            std::cout << "Unsupported activation function: " << act << "\n";
            return;
        }
	}

	std::cout << "\nCurrent model structure is valid \n";
}


