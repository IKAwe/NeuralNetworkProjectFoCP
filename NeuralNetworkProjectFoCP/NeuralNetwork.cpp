#include "NeuralNetwork.h"
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

NeuralNetwork::NeuralNetwork() {
    // Constructor can initialize weights if needed
}

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
    // For simplicity, return the first output neuron activation
    return activations.empty() ? 0.0f : activations[0];
}

/**
 * @brief  Save current weights to a file.
 * @param filename The name of the file to save the weights to.
 */

void NeuralNetwork::save_weights_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (const std::vector<std::vector<float>> layer : weights) {
        for (const std::vector<float>& neuron : layer) {
            for (float weight : neuron) {
                file << weight << ",";
            }
            file << "\n";
        }
		file << "\n fel"; // Separate layers by an empty line
    }
    file.close();
}

/**
 * @brief  Load weights from a file. (each line corresponds to a neuron's weights, each weight is separated by a comma)
 * @param filename The name of the file to load the weights from.
 */

void NeuralNetwork::load_weights_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }
    weights.clear();
    std::string line;
    std::vector<std::vector<float>> layer;
    while (std::getline(file, line)) {
        std::vector<float> neuron;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            std::string token = line.substr(0, pos);
            neuron.push_back(std::stof(token));
            line.erase(0, pos + 1);
        }
        if (!neuron.empty()) {
            layer.push_back(neuron);
        }
        else if (!layer.empty()) {
            weights.push_back(layer);
            layer.clear();
        }
    }
    if (!layer.empty()) {
        weights.push_back(layer);
    }
	std::cout << weights.size() << " layers loaded from file." << std::endl;
    for (const auto& layer : weights) {
        std::cout << "Layer " << (&layer - &weights[0]) + 1 << ":\n";
        for (const auto& neuron : layer) {
            std::cout << "  Neuron weights: ";
            for (const auto& weight : neuron) {
                std::cout << weight << " ";
            }
            std::cout << "\n";
        }
    }
    file.close();
}