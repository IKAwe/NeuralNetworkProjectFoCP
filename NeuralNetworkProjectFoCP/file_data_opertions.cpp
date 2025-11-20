#include "file_data_operations.h"
#include <vector>
#include <iostream>
/**
 * @brief Custom getline function to read a line from a file.
 * @param file Ifstream object to read from.
 * @param line String to store the read line.
 * @return Returns true if a line was read, false if end of file is reached.
 */
bool my_getline(std::ifstream& file, std::string& line) {
    line.clear();
    char ch;

    // Try to read first character
    if (!(file >> std::noskipws >> ch)) {
        return false;  // Couldn't read anything - error or end of file
    }

    // If first char is newline, return empty line
    if (ch == '\n') {
        return true;
    }

    line += ch;

    // Read rest of line
    while (file >> std::noskipws >> ch) {
        if (ch == '\n') {
            return true;
        }
        line += ch;
    }

	// Reached the end of file - return true if have written something into line
    return !line.empty();
}
/**
 * @brief Convert string line of comma-separated floats into a vector of floats.
 * @param line String line containing comma-separated float values.
 * @return One-dimensional vector of floats.
 */
std::vector<std::string> get_strings_vector_from_line(const std::string& line) {
    std::vector<std::string> result;
    std::string number_str;

    for (char ch : line) {
        if (ch == ',') {
            if (!number_str.empty()) {
                result.push_back(number_str);
                number_str.clear();
            }
        }
        else {
            number_str += ch;
        }
    }

    // Add the last number if exists
    if (!number_str.empty()) {
        result.push_back(number_str);
    }

    return result;
}

/**
 * @brief Get a vector of floats from a comma-separated line.
 * @param line String line containing comma-separated float values.
 * @return One-dimensional vector of floats.
 */

std::vector<float> get_floats_vector_from_line(const std::string& line) {
    std::vector<float> result;
    std::string number_str;

    for (char ch : line) {
        if (ch == ',') {
            if (!number_str.empty()) {
                result.push_back(std::stof(number_str));
                number_str.clear();
            }
        }
        else {
            number_str += ch;
        }
    }

    // Add the last number if exists
    if (!number_str.empty()) {
        result.push_back(std::stof(number_str));
    }

    return result;
}

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


void save_3dimensional_vector_to_file(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (const auto& matrix : vec) {
        for (const auto& row : matrix) {
            for (const auto& value : row) {
                file << value << ",";
            }
            file << "\n";
        }
        file << "\n";
    }

    file.close();
}

void load_3dimensional_vector_from_file(const std::string& filename, std::vector<std::vector<std::vector<float>>>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    // Clear vec
    vec.clear();

    std::vector<std::vector<float>> layer;
    std::string line;

    while (my_getline(file, line)) {
        if (line.empty()) {
            // Empty line - layer separator
            if (!layer.empty()) {
                vec.push_back(layer);
                layer.clear();
            }
        }
        else {
            // Parse the line directly into floats
            std::vector<float> neuron = get_floats_vector_from_line(line);
            layer.push_back(neuron);
        }
    }

    // Add the final layer if it has content
    if (!layer.empty()) {
        vec.push_back(layer);
    }

    file.close();
}