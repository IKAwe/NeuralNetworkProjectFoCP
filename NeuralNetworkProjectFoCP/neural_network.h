#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <random>

class NeuralNetwork {
public:
    NeuralNetwork();
    void initialize_weights_and_biases(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size, 
		int seed = std::random_device()(),
		std::vector<std::string> activation_functions = {});
    void visualize_model() const;
	std::vector<std::vector<float>> feedforward(const std::vector<std::vector<float>>& input) const;

	void train(const std::vector<std::vector<float>>& inputs,
				const std::vector<std::vector<float>>& targets,
				const std::vector<std::vector<float>>& test_inputs,
				const std::vector<std::vector<float>>& test_targets,
				int epochs,
				float learning_rate,
				const std::string& loss_function_name);

	float test_model(const std::vector<std::vector<float>>& test_inputs,
                 const std::vector<std::vector<float>>& test_targets,
				 const std::string& loss_function_name) const;
	void save_model_to_file(const std::string& filename) const;
	void load_model_from_file(const std::string& filename);

	// Getters (mainly for testing)
	const std::vector<std::vector<std::vector<float>>>& get_weights() const;
	const std::vector<std::vector<float>>& get_biases() const;
	const std::vector<std::string>& get_activations() const;

private:
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;
	std::vector<std::string> activations;

	bool is_model_valid = false;
	void validate_model_structure();
	
};
#endif