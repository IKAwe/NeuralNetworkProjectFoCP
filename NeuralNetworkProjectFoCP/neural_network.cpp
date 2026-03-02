#include "neural_network.h"
#include "file_data_operations.h"
#include "math_functions.h"
#include <random>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "display.h"


NeuralNetwork::NeuralNetwork() {}



void NeuralNetwork::initialize_weights_and_biases(int input_size, int hidden_layers_number, int neurons_per_hidden_layer, int output_size, std::string initialization_method,
                                                std::vector<std::string> activation_functions_passed, int seed) {
    weights.clear();
    weights.resize(hidden_layers_number + 1);

    std::default_random_engine generator(seed);

    int prev_size = input_size;
	float range_limit;

    std::unordered_map<std::string, int> initialization_methods = {
		{"Xavier", 0},
        {"He", 1}
    };
    if (initialization_methods.find(initialization_method) == initialization_methods.end()) {
        std::cerr << "Error: Unknown initialization method '" << initialization_method << "'. Using default (Xavier).\n";
        initialization_method = "Xavier";
	}
	int initialization_index = initialization_methods[initialization_method];
    // Hidden layers
    for (int i = 0; i < hidden_layers_number; ++i) {
		//init type handling
		//Xavier initialization or He initialization
        if (initialization_index == 0) {
            range_limit = std::sqrt(6.0f / (prev_size + neurons_per_hidden_layer));
        } else {
            range_limit = std::sqrt(6.0f / prev_size);
        }
        std::uniform_real_distribution<float> distribution(-range_limit, range_limit);


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
    //Init type handling
    weights[hidden_layers_number].resize(output_size);
    if (initialization_index == 0) {
        range_limit = std::sqrt(6.0f / (prev_size + output_size));
    }
    else {
        range_limit = std::sqrt(6.0f / prev_size);
    }
    //
    std::uniform_real_distribution<float> distribution(-range_limit, range_limit);
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
            biases[i][j] = 0.0f;  // Initialize biases to zero
        }
    }
    biases[hidden_layers_number].resize(output_size);
    for (int j = 0; j < output_size; ++j) {
        biases[hidden_layers_number][j] = 0.0f;  // Initialize biases to zero
	}


	// Initialize activations for each layer except input layer
    activations.clear();

    // Must match number of layers
    bool is_correct_size = (activation_functions_passed.size() == hidden_layers_number + 1);
    // Validate names
    bool all_valid = true;
    if (is_correct_size) {
        for (const auto& act : activation_functions_passed) {
            if (activation_map.find(act) == activation_map.end()) {
                all_valid = false;
				std::cerr << "Error: Unknown activation function '" << act << "'.\n";
                break;
            }
        }
    }
    else {
        std::cerr << "Error: Expected " << (hidden_layers_number + 1) << " activation functions, received " << activation_functions_passed.size() << ".\n";
    }

    if (is_correct_size && all_valid) {
        // Accept user-provided activations
        activations = activation_functions_passed;
    }
    else {
        // default
        if (!activation_functions_passed.empty()) {
            std::cerr << "Error: Invalid activation configuration. Using default.\n";
        }
        for (int i = 0; i < hidden_layers_number + 1; ++i) {
            activations.push_back("sigmoid");
		}
    }

    is_model_valid = true;

}



void NeuralNetwork::visualize_model() const {
	std::cout << "\n";
    print_header("Neural Network Weight Structure:");
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
    print_header("Neural Network Biases Structure:");
    for (size_t layer = 0; layer < biases.size(); ++layer) {
        std::cout << "\nLayer " << layer + 1 << " biases: ";
        for (size_t neuron = 0; neuron < biases[layer].size(); ++neuron) {
            std::cout << std::fixed << std::setprecision(2) << "\t" << biases[layer][neuron] << " ";
        }
        std::cout << "\n";
	}
	std::cout <<std::noshowpos;
}


std::vector<std::vector<float>> NeuralNetwork::feedforward(const std::vector<std::vector<float>>& input) const {
    if (!is_model_valid) {
        std::cout << "Model structure is invalid. Cannot perform feedforward.\n";
        return { {} };
	}
	std::vector<std::vector<float>> output(input.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)input.size(); ++i) {
        const auto& record = input[i];
        std::vector <float> prev_layer_output = record;
        for (int layer_nb = 0; layer_nb < weights.size(); ++layer_nb) {
            const auto& layer = weights[layer_nb];
            std::vector<float> curr_layer_output(layer.size());
            for (int neuron_nb = 0; neuron_nb < layer.size(); ++neuron_nb) {
                //print_vector(curr_layer_output);
                float sum = biases[layer_nb][neuron_nb];

				// z = w1*x1 + w2*x2 + ... + wn*xn + b
#pragma omp simd reduction(+:sum)
                for (int weight_nb = 0; weight_nb < layer[neuron_nb].size(); ++weight_nb) {
                    sum += layer[neuron_nb][weight_nb] * prev_layer_output[weight_nb];
                }
                curr_layer_output[neuron_nb] = sum;
            }
            // Apply activation function
            curr_layer_output = activation_map.at(activations[layer_nb]).func(curr_layer_output); //activation_map is const
            prev_layer_output = curr_layer_output;
        }
		output[i] = std::move(prev_layer_output);
    }
	return output;
}

float NeuralNetwork::test_model(const std::vector<std::vector<float>>& test_inputs,
                                const std::vector<std::vector<float>>& test_targets,
                                const std::string& loss_function_name) const{
    if (!is_model_valid) {
        std::cout << "Model structure is invalid. Cannot perform testing.\n";
        return -1.0f;
    }
    if(test_inputs.size() != test_targets.size()) {
        std::cout << "Mismatch between number of test inputs and test targets.\n";
        return -1.0f;
	}
    //Check if given vectors have valid sizes

    // Get model predictions
    const std::vector<std::vector<float>>& predictions = feedforward(test_inputs);

    // Compute and return the loss
	float sample_loss_mean = 0.0f;
    float loss = 0.0f;
#pragma omp parallel for reduction(+:loss) schedule(static)
    for (size_t i = 0; i < predictions.size(); ++i) {
		sample_loss_mean = 0.0f;
        std::vector<float> sample_loss = loss_function_map.at(loss_function_name).func(predictions[i], test_targets[i]);
        for (const auto& val : sample_loss) {
            sample_loss_mean += val;
        }
		loss+= sample_loss_mean / sample_loss.size();
	}

    return loss / predictions.size();
}


void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs,
                          const std::vector<std::vector<float>>& targets,
                            const std::vector<std::vector<float>>& test_inputs,
                            const std::vector<std::vector<float>>& test_targets ,
                          const int epochs,
                          const float learning_rate,
                          const std::string& loss_function_name) {
    print_header(" TRAINING ");
    if (!is_model_valid) {
        std::cout << "Model structure is not valid. Cannot perform training.\n";
        return;
	}

    // preallocate 
	std::vector<std::vector<float>> neuron_outputs(weights.size());// z values
    std::vector<std::vector<float>> activated_neuron_outputs(weights.size()); //h(z) vals
	std::vector<std::vector<float>> deltas(weights.size()); //gradients descents: dL/dz

	//Preallocate sizes
    for (size_t l = 0; l < weights.size(); ++l) {
        neuron_outputs[l].resize(weights[l].size());
        activated_neuron_outputs[l].resize(weights[l].size());
        deltas[l].resize(weights[l].size());
    }


    float epoch_loss;
    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
		
		epoch_loss = 0.0f;

        for (int record = 0; record < inputs.size(); ++record) {
            // ===== Forward pass - store outputs for each neuron ====
            const std::vector<float>* prev_layer_ptr = &inputs[record];

            for (int layer_nb = 0; layer_nb < weights.size(); ++layer_nb) {
                const auto& layer_weights = weights[layer_nb];

                for (int neuron_nb = 0; neuron_nb < layer_weights.size(); ++neuron_nb) {
                    float sum = biases[layer_nb][neuron_nb];

					// Weighted sum = the dot product of weights and inputs
                    for (int weight_nb = 0; weight_nb < layer_weights[neuron_nb].size(); ++weight_nb) {
                        sum += layer_weights[neuron_nb][weight_nb] * (*prev_layer_ptr)[weight_nb];
                    }
                    neuron_outputs[layer_nb][neuron_nb] = sum;
                }
				// Activation
                activated_neuron_outputs[layer_nb] = activation_map.at(activations[layer_nb]).func(neuron_outputs[layer_nb]); //activation_map is const
                prev_layer_ptr = &activated_neuron_outputs[layer_nb];
            }
			// ===== Loss calculation =====
            float record_loss_mean = 0;
			//Calc average loss for this record
			std::vector<float> record_losses = loss_function_map.at(loss_function_name).func(activated_neuron_outputs.back(), targets[record]);
            for (const auto& val : record_losses) {
                record_loss_mean += val;
			}
            record_loss_mean /= record_losses.size();
			//Accumulate epoch loss
            epoch_loss += record_loss_mean;

            // ===== Backward pass - compute gradients and update weights =====

			//Last layer delta

			// Maybe preallocate these (to do_later)
            std::vector<float> dL_dh = loss_function_map.at(loss_function_name).derivative(activated_neuron_outputs.back(), targets[record]);
            std::vector<float> dh_dz = activation_map.at(activations.back()).derivative(neuron_outputs.back()); 

			//compute dL/dh and dh/dz -  delta for last layer
            for (int i = 0; i < dh_dz.size(); i++)
                deltas.back()[i] = dL_dh[i] * dh_dz[i];

			//Starting from second last layer
            for (int layer = weights.size() - 2; layer >= 0; --layer) {
				    //dz+1/dz = dz+1/dh * dh/dz (where dz+1/dh = weights of next layer)
					// 1. Compute dh/dz for this layer - activation derivative
                    std::vector<float> dh_dz = activation_map.at(activations[layer]).derivative(neuron_outputs[layer]);

					// 2. Compute delta[layer] - dL/dz for this layer
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < dh_dz.size(); i++) {
                        float sum = 0.0f;
                        for (int j = 0; j < weights[layer + 1].size(); j++) {
                            sum += weights[layer + 1][j][i] * deltas[layer + 1][j];
                        }
                        deltas[layer][i] = sum * dh_dz[i]; //dL/dz
                    }
                }
			//===== Update weights and biases =====
            for (int layer = 0; layer < weights.size(); layer++) {
                const float* h_prev_ptr = (layer == 0) ? inputs[record].data() : activated_neuron_outputs[layer - 1].data();
#pragma omp parallel for schedule(static)
                for (int neuron = 0; neuron < weights[layer].size(); neuron++) {
                    float delta_neuron = deltas[layer][neuron];
                    float lr_delta = learning_rate * delta_neuron;
                    float* w_ptr = weights[layer][neuron].data();
                    int weight_count = (int)weights[layer][neuron].size();

#pragma omp simd
                    for (int w = 0; w < weight_count; w++) {
                        // Teraz pętla jest czysta i procesor może użyć instrukcji AVX/SSE
                        w_ptr[w] -= lr_delta * h_prev_ptr[w];
                    }

                    biases[layer][neuron] -= lr_delta;
                }
            }
        }

		//Loss calculation per epoch
        float mean_loss = epoch_loss / (inputs.size());
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Mean Loss: " << std::setprecision(4) << mean_loss << ", ";
        std::cout<< "Test Loss: " << std::setprecision(4) << test_model(test_inputs, test_targets, loss_function_name) << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Training completed in " << elapsed.count() << " seconds.\n";
}




bool NeuralNetwork::save_model_to_file(const std::string& filename) const {
    if(is_model_valid == false) {
        std::cout << "Model structure is invalid. Cannot save model to file.\n";
        return false;
	}
	std::cout << "\nSaving model to file base name - " << filename << " ...\n";
    if(save_vector_to_file(filename + "_weights.txt", weights) &&
       save_vector_to_file(filename + "_biases.txt", biases) &&
       save_vector_to_file(filename + "_activations.txt", activations)) {
        std::cout << "\nModel was successfully saved to file base name: " << filename << "\n";
		return true;
    }
    else {
		std::cerr << "\nError: Could not save model to file base name: " << filename << "\n";
		return false;
    }
}




bool NeuralNetwork::load_model_from_file(const std::string& filename) {
    std::cout << "\nTrying to load model from file base name: " << filename << " ...\n";
    if (load_vector_from_file(filename + "_weights.txt", weights) &&
        load_vector_from_file(filename + "_biases.txt", biases) &&
        load_vector_from_file(filename + "_activations.txt", activations)) {
        std::cout << "Model was successfully read from file base name: " << filename << "\n";
        validate_model_structure();
        return true;
    }
    else {
        std::cerr << "Error: Could not load model from file base name: " << filename << "\n";
        return false;
    }
}

void NeuralNetwork::validate_model_structure() {
	is_model_valid = false;
    if (weights.empty() || weights[0].empty()) {
        std::cout << "Model is empty or uninitialized.\n";
        return;
    }
	auto input_size = weights[0][0].size();
	auto hidden_layers_number = weights.size() - 1;
	auto neurons_per_hidden_layer = weights[0].size();
	auto output_size = weights[hidden_layers_number].size();


   
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
    if (activations.size() != weights.size()) {
        is_model_valid = false;
        std::cout << "Mismatch: Have " << activations.size() << " activations for " << weights.size() << " layers.\n";
        return;
    }
    
    for (const auto& act : activations) {
		if(activation_map.find(act) ==  activation_map.end()) {
            is_model_valid = false;
            std::cout << "Unknown activation function: " << act << "\n";
            return;
        }
	}
    if (activations.size() != weights.size()) {
        is_model_valid = false;
        std::cout << "Mismatch between number of activation functions and layers\n";
        return;
	}

	std::cout << "Current model structure is valid \n";
	is_model_valid = true;
}


//====================== GETTERS (mainly for testing)==========================


const std::vector<std::vector<std::vector<float>>>& NeuralNetwork::get_weights() const {
    return weights;
}


const std::vector<std::vector<float>>& NeuralNetwork::get_biases() const {
    return biases;
}

const std::vector<std::string>& NeuralNetwork::get_activations() const {
    return activations;
}


