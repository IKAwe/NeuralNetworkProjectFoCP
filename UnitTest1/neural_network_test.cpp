#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetworkProjectFoCP/neural_network.h"
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
                3, 2, 4, 1, 1234,
                { "relu", "relu", "sigmoid" }
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
			for(int i = 0; i < weights_original.size(); ++i)
			{
				for(int j = 0; j < weights_original[i].size(); ++j)
				{
					for(int k = 0; k < weights_original[i][j].size(); ++k)
					{
						Assert::AreEqual(weights_original[i][j][k], weights_loaded[i][j][k], 0.0001f);
					}
				}
			}
			// Compare biases
			for(int i = 0; i < biases_original.size(); ++i)
			{
				for(int j = 0; j < biases_original[i].size(); ++j)
				{
					Assert::AreEqual(biases_original[i][j], biases_loaded[i][j], 0.0001f);
				}
			}
			// Compare activations
			for(int i = 0; i < activations_original.size(); ++i)
			{
				Assert::AreEqual(activations_original[i], activations_loaded[i]);
			}


		}

		TEST_METHOD(feedforward_test)
		{
			NeuralNetwork nn;
			nn.initialize_weights_and_biases(
				2, 1, 2, 1, 42,
				{ "sigmoid", "sigmoid" }
			);
			std::vector<std::vector<float>> input = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
			auto output = nn.feedforward(input);
			// Since weights are initialized randomly, we cant predict exact outputs
			// Check the dimensions of the output
			Assert::AreEqual((int)(output.size()), 4); // 4 input samples
			Assert::AreEqual((int)(output[0].size()), 1); // 1 output neuron

			//check if it returns the same output for same input
			auto output2 = nn.feedforward(input);
			for (size_t i = 0; i < output.size(); ++i)
			{
				Assert::AreEqual(output[i][0], output2[i][0], 0.0001f);
			}

		}
	};
}
