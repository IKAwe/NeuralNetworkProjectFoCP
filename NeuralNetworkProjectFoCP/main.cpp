#include <vector>
#include <iostream>
#include "csv_parser.h"
#include "NeuralNetwork.h"

void getInputData(int header[], int input_size, int& input_data_array)
{
    for (int i = 0; i < input_size; i++) {
		std::cout << header[i] << ": ";
		std::cin >> header[i];
    }
}

int main()
{
    //int* p = x;
    auto data = parseCSV("iris.csv");
    std::vector<int> input_data;

    /*for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
	std::cout << data.size() << std::endl << data[0].size();*/
	NeuralNetwork nn;
    nn.initialize_weights(2, 2, 4, 3);
    nn.visualize_weights();
    return 0;
}
