#include "neural_network.h"
#include "file_data_operations.h"
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::initialize_weights(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size) {
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
}

void NeuralNetwork::visualize_weights() const {
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
}

float NeuralNetwork::feedforward(const std::vector<float>& input) const {
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

void NeuralNetwork::save_weights_to_file(const std::string& filename) const {
    save_3dimensional_vector_to_file(filename, weights, activations, true);
}



/**
 * @brief  Load weights from a file. (each line corresponds to a neuron's weights, each weight is separated by a comma)
 * @param filename The name of the file to load the weights from.
 */

void NeuralNetwork::load_weights_from_file(const std::string& filename) {
    load_3dimensional_vector_from_file(filename, weights, activations, true);
}


