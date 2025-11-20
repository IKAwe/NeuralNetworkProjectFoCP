#include "parsing_files.h"
#include <vector>


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
        return false;  // Couldn't read anything (EOF or error)
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

    // Reached EOF - return true if we read something
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
