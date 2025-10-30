#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();
    void initialize_weights(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size);
    void visualize_weights() const;

private:
    std::vector<std::vector<std::vector<float>>> weights;
};

#endif
