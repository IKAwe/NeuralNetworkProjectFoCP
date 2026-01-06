#include "file_data_operations.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>


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

    // Add the last number if exists (when there's no comma at the end)
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
std::vector<std::vector<std::string>>parseCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<std::string>> data;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    // Read data rows
	int row_el_nb = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row = get_strings_from_line(line);
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;  // Changed
}

/**
 * @brief  Save 3-dimensional floats vector to a txt file - separated by newlines and commas
 * @param filename Name of the file to write the vector to
 * @param vec Vector to be saved
 */

 // 1D vector overload
void save_vector_to_file(const std::string& filename,const std::vector<float>& vec,char delimiter) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    // Write 1D data
    if (!vec.empty()) {
        file << vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            file << delimiter << vec[i];
        }
        file << "\n";
    }
}
void save_vector_to_file(const std::string& filename, const std::vector<std::string>& vec, char delimiter) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    // Write 1D data
    if (!vec.empty()) {
        file << vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            file << delimiter << vec[i];
        }
        file << "\n";
    }
}
// 2D vector overload
void save_vector_to_file(const std::string& filename,const std::vector<std::vector<float>>& vec,char delimiter) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write 2D data (no empty lines between rows)
    for (const auto& row : vec) {
        if (!row.empty()) {
            file << row[0];
            for (size_t i = 1; i < row.size(); ++i) {
                file << delimiter << row[i];
            }
        }
        file << "\n";
    }
}

// 3D vector overload
void save_vector_to_file(const std::string& filename,const std::vector<std::vector<std::vector<float>>>& vec,char delimiter) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write 3D data (separate matrices with empty lines)
    for (size_t matrix_idx = 0; matrix_idx < vec.size(); ++matrix_idx) {
        const auto& matrix = vec[matrix_idx];

        for (const auto& row : matrix) {
            if (!row.empty()) {
                file << row[0];
                for (size_t i = 1; i < row.size(); ++i) {
                    file << delimiter << row[i];
                }
            }
            file << "\n";
        }
        // Add empty line between matrices, but not after the last one
        if (matrix_idx < vec.size() - 1) {
            file << "\n";
        }
    }
}

/**
 * @brief  Load 3-dimensional floats vector from a file.
 * @param filename The name of the file to load the vector from.
 * @param vec The vector to load the data into.
 */


 // 1D vector overload for strings
void load_vector_from_file(const std::string& filename,
    std::vector<std::string>& vec,
    char delimiter) {
    vec.clear();
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                vec.push_back(cell);
            }
        }
    }
}

// 1D vector overload for floats
void load_vector_from_file(const std::string& filename,
    std::vector<float>& vec,
    char delimiter) {
    vec.clear();
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                try{
                    vec.push_back(std::stof(cell));
                }
                catch (...) {
                    std::cerr << "Error: Could not convert cell to float: " << cell << std::endl;
				}
                //vec.push_back(std::stof(cell));
            }
        }
    }
}

// 2D vector overload
void load_vector_from_file(const std::string& filename,std::vector<std::vector<float>>& vec,char delimiter) {
    vec.clear();
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                try{
                    row.push_back(std::stof(cell));
                }
                catch (...) {
                    std::cerr << "Error: Could not convert cell to float: " << cell << std::endl;
                }
                //row.push_back(std::stof(cell));
            }
        }

        if (!row.empty()) {
            vec.push_back(row);
        }
    }
}

// 3D vector overload
void load_vector_from_file(const std::string& filename,std::vector<std::vector<std::vector<float>>>& vec,char delimiter) {
    vec.clear();
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::vector<std::vector<float>> current_matrix;
    std::string line;

    while (std::getline(file, line)) {
        // Empty line indicates matrix separation
        if (line.empty()) {
            if (!current_matrix.empty()) {
                vec.push_back(current_matrix);
                current_matrix.clear();
            }
            continue;
        }

        // Parse non-empty line as a row
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                try{
                    row.push_back(std::stof(cell));
                }
                catch(...){
                    std::cerr << "Error: Could not convert cell to float: " << cell << std::endl;
				}
                //row.push_back(std::stof(cell));
            }
        }

        if (!row.empty()) {
            current_matrix.push_back(row);
        }
    }

    // Don't forget the last matrix
    if (!current_matrix.empty()) {
        vec.push_back(current_matrix);
    }
}


int get_index(const std::vector<std::string>& vec,const std::string& value) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == value) {
            return i;
        }
    }
    return -1; // Not found
}
