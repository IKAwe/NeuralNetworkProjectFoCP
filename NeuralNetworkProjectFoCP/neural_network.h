#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
/** @file neural_network.h
 * @brief Core Neural Network class definition and lifecycle management.
 */
#include <vector>
#include <string>
#include <random>

 /**
  * @class NeuralNetwork
  * @brief Implementation of a multilayer perceptron with backpropagation capabilities.
  */
class NeuralNetwork {
public:
    NeuralNetwork();
	/**
	 * @brief Initialize weights and biases for the neural network.
	 * @details Supports Xavier and He initialization methods. Biases are initialized to zero.
	 * @param input_size The size of the input layer.
	 * @param hidden_layers_number The number of hidden layers.
	 * @param neurons_per_hidden_layer The number of neurons in each hidden layer.
	 * @param output_size The size of the output layer.
	 * @param initialization_method The method used ("Xavier" or "He"). Defaults to Xavier if unknown.
	 * @param activation_functions_passed Activation functions for each layer (count should be hidden_layers_number + 1).
	 * @param seed The random seed for reproducibility.
	 */
	void initialize_weights_and_biases(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size, std::string initialization_method = "Xavier",
		std::vector<std::string> activation_functions = {},
		int seed = std::random_device()());
	/**
	 * @brief Visualize the neural network model structure (weights and biases).
	 */
    void visualize_model() const;
	/**
	 * @brief Performs a forward pass through the network for multiple input samples.
	 * @param input A 2D vector where each row is an input sample.
	 * @return A 2D vector of predictions. Returns an empty 2D vector if the model is invalid.
	 * @note Ensure the input sample size matches the network's input layer size.
	 */
	std::vector<std::vector<float>> feedforward(const std::vector<std::vector<float>>& input) const;
	/**
	 * @brief Train the neural network using the provided training data.
	 * @param inputs The input data for training - each row represents a training sample. It is expected to be shuffled.
	 * @param targets The target output data for training - each row represents the target output for the corresponding training sample.
	 * @param test_inputs The input data for testing during training - each row represents a different test sample.
	 * @param test_targets The target output data for testing during training - each row represents the target output for the corresponding test sample.
	 * @param epochs The number of epochs to train for.
	 * @param learning_rate The learning rate for weight updates.
	 * @param loss_function_name The name of the loss function to use - right now only "MSE".
	 */
	void train(const std::vector<std::vector<float>>& inputs,
				const std::vector<std::vector<float>>& targets,
				const std::vector<std::vector<float>>& test_inputs,
				const std::vector<std::vector<float>>& test_targets,
				int epochs,
				float learning_rate,
				const std::string& loss_function_name);
	/**
	 * @brief Calculates the network loss on a test dataset.
	 * @param test_inputs Input samples for testing.
	 * @param test_targets Expected output labels.
	 * @param loss_function_name The name of the loss function (e.g., "MSE").
	 * @return The average loss across all samples. Returns -1.0f if the model is invalid or sizes mismatch.
	 */
	float test_model(const std::vector<std::vector<float>>& test_inputs,
                 const std::vector<std::vector<float>>& test_targets,
				 const std::string& loss_function_name) const;
	/**
	 * @brief Saves the model state (weights, biases, activations) to 3 files: {filename}_weights.txt, {filename}_biases.txt, {filename}_activations.txt.
	 * @param filename The base name of the files (appends _weights.txt, etc. automatically).
	 */
	void save_model_to_file(const std::string& filename) const;
	/**
	 * @brief Loads the model state (weights, biases, activations) from disk.
	 * @param filename The base name of the files (appends _weights.txt, etc. automatically).
	 * @note This function triggers validate_model_structure() after loading.
	 */
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