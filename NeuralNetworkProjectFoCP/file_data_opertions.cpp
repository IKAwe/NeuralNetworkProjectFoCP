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

    // Trying to read first character
    if (!(file >> std::noskipws >> ch)) {
        return false;  // Read nothing - error or end of file
    }

    // If first character is \n, return empty line
    if (ch == '\n') {
        return true;
    }

    line += ch;

    // Read the line
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
std::vector<std::string> get_strings_from_line(const std::string& line) {
    std::vector<std::string> result;
    std::string str;

    for (char ch : line) {
        if (ch == ',') {
            if (!str.empty()) {
                result.push_back(str);
                str.clear();
            }
        }
        else {
            str += ch;
        }
    }

    // Add the last number if exists
    if (!str.empty()) {
        result.push_back(str);
    }

    return result;
}

/**
 * @brief Get a vector of floats from a comma-separated line.
 * @param line String line containing comma-separated float values.
 * @return One-dimensional vector of floats.
 */

std::vector<float> get_floats_from_line(const std::string& line) {
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

	// Add the last number if exists (when there's no  comma at the end)
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

    bool isHeader = true; // Skip first row

    while (my_getline(file, line)) {
        if (isHeader) {
            // Skip header row
            isHeader = false;
            continue;
        }

        std::vector<std::string> row = get_strings_from_line(line);
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

/**
 * @brief  Save 3-dimensional floats vector to a txt file - separated by newlines and commas
 * @param filename Name of the file to write the vector to
 * @param vec Vector to be saved
 */
void save_3dimensional_vector_to_file(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec, const std::vector<std::string>& header, bool writeHeader) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header as first line if provided
    if (writeHeader && !header.empty()) {
        for (int i = 0; i < header.size(); ++i) {
            file << header[i];
            if (i < header.size() - 1) file << ",";
        }
    }
    file << "\n";


	// Write each 2-dimensional matrix separated by an empty line
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

/**
 * @brief  Load 3-dimensional floats vector from a file.
 * @param filename The name of the file to load the vector from.
 * @param vec The vector to load the data into.
 */
void load_3dimensional_vector_from_file(const std::string& filename, std::vector<std::vector<std::vector<float>>>& vec, std::vector<std::string>& header, bool isThereHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: can't open file for reading: " << filename << std::endl;
        return;
    }
    vec.clear();
    header.clear();

    std::vector<std::vector<float>> layer;
    std::string line;

    while (my_getline(file, line)) {
        if (isThereHeader) {
            // First line is header
            header = get_strings_from_line(line);
            isThereHeader = false;
            continue;
        }
        if (line.empty()) {
            // Empty line - layer separator
            if (!layer.empty()) {
                vec.push_back(layer);
                layer.clear();
            }
        }
        else {
            // Parse the line into floats vec
            std::vector<float> neuron = get_floats_from_line(line);
            layer.push_back(neuron);
        }
    }

    // Add the final layer if it has content
    if (!layer.empty()) {
        vec.push_back(layer);
    }

    file.close();
}