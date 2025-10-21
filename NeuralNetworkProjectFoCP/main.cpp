#include <vector>
#include <iostream>
#include "csv_parser.h"
int main()
{
    const int a = 14, b = 10;
    int x[a][b]{};
    //int* p = x;
   /* auto data = parseCSV("iris.csv");

    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }*/

    for (int i = 0; i < 100; ++i) {
        std::cout << x[14][i] << std::endl;

	}
    return 0;
}
