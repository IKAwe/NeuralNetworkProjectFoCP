#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetworkProjectFoCP/neural_network.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest1
{
	TEST_CLASS(NeuralNetworkTest)
	{
	public:
		
		TEST_METHOD(saving_loading_test)
		{
			NeuralNetwork nn;
				nn.initialize_weights_and_biases(3, 2, 4, 1, 1234, { "relu", "relu", "sigmoid" });
				nn.save_model_to_file("test_model.nn");
				NeuralNetwork nn_loaded;
				nn_loaded.load_model_from_file("test_model.nn");
				// Compare weights
				const auto& weights_original = nn.get_weights();
				const auto& weights_loaded = nn_loaded.get_weights();

		}
	};
}
