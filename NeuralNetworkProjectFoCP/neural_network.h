#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork();
    void initialize_weights(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size);
    void visualize_weights() const;
	float feedforward(const std::vector<float>& input) const;

	void save_weights_to_file(const std::string& filename) const;
	void load_weights_from_file(const std::string& filename);

private:
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;
	std::vector<std::string> activations = {"sigmoid", "relu", "relu"};
};
#endif