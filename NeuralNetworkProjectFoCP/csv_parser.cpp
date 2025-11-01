#include "csv_parser.h"
#include <fstream>
#include <sstream>
#include <vector>



#include <string>

#include <iostream>

std::vector<std::vector<float>> parseCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<float>> data;

    bool isHeader = true; // Flag to skip first row

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        if (isHeader) {
            // Skip header row
            isHeader = false;
            continue;
        }

        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            try {
                float value = std::stof(cell);
                row.push_back(value);
            }
            catch (const std::invalid_argument&) {
                // Ignore non-numeric cells
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    return data;
}