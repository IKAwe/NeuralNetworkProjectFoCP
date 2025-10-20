#include "csv_parser.h"
#include <fstream>
#include <sstream>
#include <vector>

std::vector<std::vector<std::string>> parseCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<std::string>> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    return data;
}