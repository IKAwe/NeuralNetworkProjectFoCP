#pragma once
class NeuralNetwork
{
public:
	NeuralNetwork();
	void initialize_weights(int input_size, int hidden_layers_number, int output_size);
	void load_weights();

	void forwardPass();
	void train();
	void predict();
};

