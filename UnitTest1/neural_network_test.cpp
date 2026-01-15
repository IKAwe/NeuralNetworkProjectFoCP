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

            // Compare weights
            Assert::IsTrue(nn.get_weights().size() == nn_loaded.get_weights().size());


		}
	};
}
