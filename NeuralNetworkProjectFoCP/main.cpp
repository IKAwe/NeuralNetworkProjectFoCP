#include <vector>
#include <iostream>
#include "csv_parser.h"
int main()
{
    //int* p = x;
    auto data = parseCSV("iris.csv");

    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
