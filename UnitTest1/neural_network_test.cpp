#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetworkProjectFoCP/neural_network.h"
#include "../NeuralNetworkProjectFoCP/data_preprocessor.h"
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

			remove("test_model_weights");
			remove("test_model_biases");
			remove("test_model_activations");

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
}