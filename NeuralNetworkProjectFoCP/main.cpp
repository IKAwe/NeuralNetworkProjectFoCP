#include <vector>
#include <iostream>
#include "csv_parser.h"
#include <typeinfo>
int main()
{

    auto data = parseCSV("iris.csv");

    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
    //std::cout << typeid(data).name();
}
