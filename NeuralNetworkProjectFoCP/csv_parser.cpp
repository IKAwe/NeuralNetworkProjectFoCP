#include "csv_parser.h"
#include "parsing_files.h"
#include <fstream>
#include <vector>

#include <string>

#include <iostream>
/**
 * @brief  Loading CSV file and parsing numeric data while ignoring the header row.
 * @param filename The name of the CSV file to parse.
 * @return A 2D vector containing the parsed numeric data.
 */
std::vector<std::vector<std::string>> parseCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<std::string>> data;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    bool isHeader = true; // Flag to skip first row

    while (my_getline(file, line)) {
        if (isHeader) {
            // Skip header row
            isHeader = false;
            continue;
        }

        std::vector<std::string> row = get_strings_vector_from_line(line);
        if (!row.empty()) {
            data.push_back(row);
        }
    }
	file.close();
	return data;
}