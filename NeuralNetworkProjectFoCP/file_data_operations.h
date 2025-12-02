#pragma once
#ifndef FILE_DATA_OPERATIONS_H
#define FILE_DATA_OPERATIONS_H

#include <vector>
#include <string>
#include <fstream>

//Parsing functions
std::vector<std::string> get_strings_from_line(const std::string& line);
std::vector<float> get_floats_from_line(const std::string& line);

//Loading/saving to file functions
std::pair<std::vector<std::string>, std::vector<std::vector<std::string>>> parseCSV(const std::string& filename);

// Overload for 1D vectors
void save_vector_to_file(const std::string& filename,
    const std::vector<float>& vec,
    char delimiter = ',');
void save_vector_to_file(const std::string& filename, const std::vector<std::string>& vec, char delimiter = ',');
// Overload for 2D vectors  
void save_vector_to_file(const std::string& filename,
    const std::vector<std::vector<float>>& vec,
    char delimiter = ',');
// Overload for 3D vectors
void save_vector_to_file(const std::string& filename,const std::vector<std::vector<std::vector<float>>>& vec,char delimiter = ',');

void load_vector_from_file(const std::string& filename, std::vector<std::string>& vec, char delimiter = ',');
void load_vector_from_file(const std::string& filename, std::vector<std::vector<float>>& vec, char delimiter = ',');
void load_vector_from_file(const std::string& filename, std::vector<std::vector<std::vector<float>>>& vec, char delimiter = ',');
    ;
#endif