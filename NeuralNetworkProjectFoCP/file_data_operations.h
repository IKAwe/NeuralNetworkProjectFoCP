#pragma once
#ifndef FILE_DATA_OPERATIONS_H
#define FILE_DATA_OPERATIONS_H

#include <vector>
#include <string>
#include <fstream>


//Parsing functions
bool my_getline(std::ifstream& file, std::string& line);
std::vector<std::string> get_strings_from_line(const std::string& line);
std::vector<float> get_floats_from_line(const std::string& line);

//Loading/saving to file functions
std::vector<std::vector<std::string>> parseCSV(const std::string& filename);

void save_3dimensional_vector_to_file(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec, const std::vector<std::string>& header = {}, bool writeHeader = false);
void load_3dimensional_vector_from_file(const std::string& filename, std::vector<std::vector<std::vector<float>>>& vec, std::vector<std::string>& header, bool isThereHeader = false);
#endif