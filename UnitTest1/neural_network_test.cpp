#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetworkProjectFoCP/neural_network.h"
#include "../NeuralNetworkProjectFoCP/data_preprocessor.h"
#include "../NeuralNetworkProjectFoCP/file_data_operations.h"
#include <iostream>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace neural_network_test
{
	TEST_CLASS(NeuralNetworkTest)
	{
	public:

		TEST_METHOD(saving_loading_test)
		{
			NeuralNetwork nn;
			nn.initialize_weights_and_biases(
				3, 2, 4, 1, "Xavier",
				{ "tanh", "relu", "sigmoid" }, 1234
			);

			nn.save_model_to_file("test_model");

			NeuralNetwork nn_loaded;
			nn_loaded.load_model_from_file("test_model");

			remove("test_model_weights.txt");
			remove("test_model_biases.txt");
			remove("test_model_activations.txt");

			auto& weights_original = nn.get_weights();
			auto& weights_loaded = nn_loaded.get_weights();

			auto& biases_original = nn.get_biases();
			auto& biases_loaded = nn_loaded.get_biases();

			auto& activations_original = nn.get_activations();
			auto& activations_loaded = nn_loaded.get_activations();

			// Compare weights
			for (int i = 0; i < weights_original.size(); ++i)
			{
				for (int j = 0; j < weights_original[i].size(); ++j)
				{
					for (int k = 0; k < weights_original[i][j].size(); ++k)
					{
						Assert::AreEqual(weights_original[i][j][k], weights_loaded[i][j][k], 0.0001f);
					}
				}
			}
			// Compare biases
			for (int i = 0; i < biases_original.size(); ++i)
			{
				for (int j = 0; j < biases_original[i].size(); ++j)
				{
					Assert::AreEqual(biases_original[i][j], biases_loaded[i][j], 0.0001f);
				}
			}
			// Compare activations
			for (int i = 0; i < activations_original.size(); ++i)
			{
				Assert::AreEqual(activations_original[i], activations_loaded[i]);
			}


		}

		// Just check if the dimensions are correct after initialization
		TEST_METHOD(initialization_test)
		{
			NeuralNetwork nn;
			int input_size = 4;
			int hidden_layers = 2;
			int neurons_per_hidden = 5;
			int output_size = 3;

			// Initialize the network with 2 hidden layers and 1 output layer [cite: 143, 192]
			nn.initialize_weights_and_biases(
				input_size,
				hidden_layers,
				neurons_per_hidden,
				output_size,
				"Xavier",//doesnt matter
				{ "relu", "relu", "sigmoid" } //doesnt matter
			);

			auto& weights = nn.get_weights(); 
			auto& biases = nn.get_biases(); 
			auto& activations = nn.get_activations(); 

			// 1. Check layer count
			// Total layers should be hidden_layers + 1 (for the output layer) 
			size_t expected_layers = (size_t)hidden_layers + 1;
			Assert::AreEqual(expected_layers, weights.size());
			Assert::AreEqual(expected_layers, biases.size());
			Assert::AreEqual(expected_layers, activations.size());

			// 2. Check dimensions of the first layer (input -> first hidden)
			// Weights should be organized as [neurons_in_current_layer][neurons_in_previous_layer] 
			Assert::AreEqual((size_t)neurons_per_hidden, weights[0].size());
			Assert::AreEqual((size_t)input_size, weights[0][0].size());
			Assert::AreEqual((size_t)neurons_per_hidden, biases[0].size());

			// 3. Check dimensions of the output Layer
			// The last layer's weights should connect the last hidden layer to the output neurons 
			size_t last_idx = expected_layers - 1;
			Assert::AreEqual((size_t)output_size, weights[last_idx].size());
			Assert::AreEqual((size_t)neurons_per_hidden, weights[last_idx][0].size());
			Assert::AreEqual((size_t)output_size, biases[last_idx].size());
		}
		TEST_METHOD(feedforward_test)
		{
			NeuralNetwork nn;
			nn.load_model_from_file("model_for_testing");
			if (nn.get_activations().size() == 0)
			{
				Assert::Fail(L"Failed to load model for testing feedforward.");
			}
			std::vector<std::vector<float>> inputs = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
			std::vector<std::vector<float>> outputs = nn.feedforward(inputs);
			std::vector<std::vector<float>> expected_outputs = { {0.98201379f}, {0.98201379f}, {0.9999832985f}, {0.999664649869f} };
			for (int i = 0; i < outputs.size(); ++i)
			{
				for (int j = 0; j < outputs[i].size(); ++j)
				{
					Assert::AreEqual(expected_outputs[i][j], outputs[i][j], 0.0001f);
				}
			}


		}

		TEST_METHOD(training_test)
		{
			NeuralNetwork nn;
			nn.initialize_weights_and_biases(2, 1, 2, 1, "He", { "relu", "sigmoid" }, 42);
			std::vector<std::vector<float>> inputs = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
			std::vector<std::vector<float>> targets = { {0.0f}, {1.0f}, {1.0f}, {0.0f} };
			auto initial_results = nn.test_model(inputs, targets, "MSE");
			nn.train(inputs, targets, inputs, targets, 10000, 0.1f, "MSE");
			auto post_training_results = nn.test_model(inputs, targets, "MSE");
			Assert::IsTrue(post_training_results < initial_results);

		}
	};

	TEST_CLASS(DataPreprocessorTest)
	{
	public:
		TEST_METHOD(fit_and_transform_test)
		{
			DataPreprocessor dp;

			// Create a small dataset: Header + 2 rows
			// Col 0: Numeric, Col 1: Categorical (Target)
			std::vector<std::vector<std::string>> dataset = {
			{"feature", "label"},
			{"10.0", "A"},
				{"20.0", "B"}
			};

			// 1. Test Fitting 
			dp.fit(dataset);

			// 2. Test Transformation 
			// We want to extract "label" as our target
			auto transformed = dp.transform_and_extract(dataset, "label", true);

			auto& features = transformed.first; // Transformed feature matrix
			auto& targets = transformed.second;  // Transformed target matrix

			// Check if features were normalized (Min-Max)
			// 10.0 should become 0.0, 20.0 should become 1.0
			Assert::AreEqual(0.0f, features[0][0], 0.0001f);
			Assert::AreEqual(1.0f, features[1][0], 0.0001f);

			// Check One-Hot Encoding for categories 
			Assert::AreEqual((size_t)2, targets[0].size());
			// Sample 0 (label A) should have 1 at index 0 and 0 at index 1
			Assert::AreEqual(1.0f, targets[0][0]);
			Assert::AreEqual(0.0f, targets[0][1]);
		}

	};
	TEST_CLASS(FileDataOperationsTest) {
		public:
		TEST_METHOD(csv_parsing_test)
		{
			// Prepare a sample CSV
			const std::string csv_content = "col1,col2,col3\n1.0,2.0,3.0\n4.0,5.0,6.0\n";
			// Write to a temp
			const std::string temp_filename = "csv_parse_test_file.csv";
			std::ofstream temp_file(temp_filename);
			temp_file << csv_content;
			temp_file.close();
			// Parse the CSV file
			auto parsed_data = parseCSV(temp_filename);
			// Clean up the temp
			remove(temp_filename.c_str());
			// Validate the parsed data
			Assert::AreEqual(size_t(3), parsed_data.size()); // 3 rows including header
			Assert::AreEqual(size_t(3), parsed_data[0].size()); // 3 columns
			// Check header
			Assert::AreEqual(std::string("col1"), parsed_data[0][0]);
			Assert::AreEqual(std::string("col2"), parsed_data[0][1]);
			Assert::AreEqual(std::string("col3"), parsed_data[0][2]);
			// Check first data row
			Assert::AreEqual(std::string("1.0"), parsed_data[1][0]);
			Assert::AreEqual(std::string("2.0"), parsed_data[1][1]);
			Assert::AreEqual(std::string("3.0"), parsed_data[1][2]);
			// Check second data row
			Assert::AreEqual(std::string("4.0"), parsed_data[2][0]);
			Assert::AreEqual(std::string("5.0"), parsed_data[2][1]);
			Assert::AreEqual(std::string("6.0"), parsed_data[2][2]);
		}

		TEST_METHOD(vector_loading_saving_test)
		{
			const std::string test_1d_file = "test_1d.txt";
			const std::string test_2d_file = "test_2d.txt";
			const std::string test_3d_file = "test_3d.txt";

			// 1. Test 1D (strings - activation funcs)
			std::vector<std::string> original_1d = { "one", "two", "three" };
			Assert::IsTrue(save_vector_to_file(test_1d_file, original_1d, ','));

			std::vector<std::string> loaded_1d;
			Assert::IsTrue(load_vector_from_file(test_1d_file, loaded_1d, ','));
			Assert::AreEqual(original_1d.size(), loaded_1d.size());
			for (size_t i = 0; i < original_1d.size(); ++i) {
				Assert::AreEqual(original_1d[i], loaded_1d[i]);
			}

			// 2. Test 2D  (floats - biases)
			std::vector<std::vector<float>> original_2d = { {0.1f, 0.2f}, {0.3f, 0.4f} };
			Assert::IsTrue(save_vector_to_file(test_2d_file, original_2d, ','));

			std::vector<std::vector<float>> loaded_2d;
			Assert::IsTrue(load_vector_from_file(test_2d_file, loaded_2d, ','));
			Assert::AreEqual(original_2d.size(), loaded_2d.size());
			for (size_t i = 0; i < original_2d.size(); ++i) {
				for (size_t j = 0; j < original_2d[i].size(); ++j) {
					Assert::AreEqual(original_2d[i][j], loaded_2d[i][j], 0.0001f);
				}
			}

			// 3. Test 3D  (floats - weights)
			// [layer][[neuron][input_neuron]]
			std::vector<std::vector<std::vector<float>>> original_3d = {
				{ {0.11f, 0.12f}, {0.21f, 0.22f} },
				{ {0.55f, 0.66f} }
			};
			Assert::IsTrue(save_vector_to_file(test_3d_file, original_3d, ','));

			std::vector<std::vector<std::vector<float>>> loaded_3d;
			Assert::IsTrue(load_vector_from_file(test_3d_file, loaded_3d, ','));
			Assert::AreEqual(original_3d.size(), loaded_3d.size());

			// Cleanup files
			remove(test_1d_file.c_str());
			remove(test_2d_file.c_str());
			remove(test_3d_file.c_str());
		}

	};
}