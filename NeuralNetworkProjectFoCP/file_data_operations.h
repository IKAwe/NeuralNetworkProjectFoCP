#pragma once
#ifndef FILE_DATA_OPERATIONS_H
#define FILE_DATA_OPERATIONS_H
/** @file file_data_operations.h
 *  @brief Declaration of functions for parsing, loading, and saving data to files.
 */ 
#include <vector>
#include <string>
#include <fstream>

//Parsing functions
/**
 * @brief Convert a line of comma-separated values into a vector of strings.
 * @param line String line containing comma-separated values.
 * @return One-dimensional vector of strings.
 */
std::vector<std::string> get_strings_from_line(const std::string& line);
/**
 * @brief Get a vector of floats from a comma-separated line.
 * @param line String line containing comma-separated float values.
 * @return One-dimensional vector of floats.
 */
std::vector<float> get_floats_from_line(const std::string& line);

//Loading/saving to file functions
/**
 * @brief Loading CSV file and parsing data into a string matrix.
 * @details Note: This implementation currently reads all lines, including the header.
 * @param filename The name of the CSV file to parse.
 * @return A 2D vector (rows and columns) of strings.
 */
std::vector<std::vector<std::string>> parseCSV(const std::string& filename);

// Overload for 1D vectors
/**
 * @brief Save a 1D vector of floats to a file.
 * @param filename Name of the file to create or overwrite.
 * @param vec The 1D vector containing float data.
 * @param delimiter The character used to separate values (default is comma).
 */
void save_vector_to_file(const std::string& filename,
    const std::vector<float>& vec,
    char delimiter = ',');
/**
 * @brief Save a 1D vector of strings to a file.
 * @overload
 */
void save_vector_to_file(const std::string& filename, const std::vector<std::string>& vec, char delimiter = ',');
// Overload for 2D vectors  
/**
 * @brief Save a 2D vector (matrix) of floats to a file.
 * @details Each inner vector is written as a single line, with elements separated by the delimiter.
 * @param vec The 2D vector (matrix) of floats.
 * @overload
 */
void save_vector_to_file(const std::string& filename,
    const std::vector<std::vector<float>>& vec,
    char delimiter = ',');
// Overload for 3D vectors
/**
 * @brief Save a 3D vector (list of matrices) of floats to a file.
 * @details Matrices are separated by a blank line. Within each matrix, rows are
 * separated by newlines and elements by the delimiter.
 * @param vec The 3D vector representing multiple layers/matrices.
 * @overload
 */
void save_vector_to_file(const std::string& filename,const std::vector<std::vector<std::vector<float>>>& vec,char delimiter = ',');

// Overloads for loading from file
/**
 * @brief Load a 1D vector of strings from a single-line file.
 * @param filename The name of the file to read.
 * @param[out] vec The string vector to be populated (cleared before loading).
 * @param delimiter The character used to split the values.
 */
void load_vector_from_file(const std::string& filename, std::vector<std::string>& vec, char delimiter = ',');
/**
 * @brief Load a 1D vector of floats from a single-line file. NOT used i think
 * @details Values are converted from strings to floats. Invalid conversions are caught and logged.
 * @overload
 */
void load_vector_from_file(const std::string& filename, std::vector<float>& vec, char delimiter = ',');
// 2D vector overload
/**
 * @brief Load a 2D vector (matrix) of floats from a multi-line file.
 * @details Each non-empty line in the file is treated as a row in the matrix.
 * @param[out] vec The 2D vector to be populated.
 * @overload
 */
void load_vector_from_file(const std::string& filename, std::vector<std::vector<float>>& vec, char delimiter = ',');
// 3D vector overload
/**
 * @brief Load a 3D vector (list of matrices) from a file.
 * @details Reconstructs the 3D structure by using empty lines as separators between different matrices.
 * @param[out] vec The 3D vector to be populated.
 * @overload
 */
void load_vector_from_file(const std::string& filename, std::vector<std::vector<std::vector<float>>>& vec, char delimiter = ',');


#endif