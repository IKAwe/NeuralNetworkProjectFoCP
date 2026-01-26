#include "file_data_operations.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>


/**
 * @brief Convert a line of comma-separated values into a vector of strings.
 * @param line String line containing comma-separated values.
 * @return One-dimensional vector of strings.
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
 * @brief Loading CSV file and parsing data into a string matrix.
 * @details Note: This implementation currently reads all lines, including the header.
 * @param filename The name of the CSV file to parse.
 * @return A 2D vector (rows and columns) of strings.
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
/**
 * @brief Save a 1D vector of floats to a file.
 * @param filename Name of the file to create or overwrite.
 * @param vec The 1D vector containing float data.
 * @param delimiter The character used to separate values (default is comma).
 */
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
/**
 * @brief Save a 1D vector of strings to a file.
 * @overload
 */
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
/**
 * @brief Save a 2D vector (matrix) of floats to a file.
 * @details Each inner vector is written as a single line, with elements separated by the delimiter.
 * @param vec The 2D vector (matrix) of floats.
 * @overload
 */
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
/**
 * @brief Save a 3D vector (list of matrices) of floats to a file.
 * @details Matrices are separated by a blank line. Within each matrix, rows are
 * separated by newlines and elements by the delimiter.
 * @param vec The 3D vector representing multiple layers/matrices.
 * @overload
 */
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
/**
 * @brief Load a 1D vector of strings from a single-line file.
 * @param filename The name of the file to read.
 * @param[out] vec The string vector to be populated (cleared before loading).
 * @param delimiter The character used to split the values.
 */
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
/**
 * @brief Load a 1D vector of floats from a single-line file.
 * @details Values are converted from strings to floats. Invalid conversions are caught and logged.
 * @overload
 */
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
            }
        }
    }
}

// 2D vector overload
/**
 * @brief Load a 2D vector (matrix) of floats from a multi-line file.
 * @details Each non-empty line in the file is treated as a row in the matrix.
 * @param[out] vec The 2D vector to be populated.
 * @overload
 */
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
/**
 * @brief Load a 3D vector (list of matrices) from a file.
 * @details Reconstructs the 3D structure by using empty lines as separators between different matrices.
 * @param[out] vec The 3D vector to be populated.
 * @overload
 */
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

