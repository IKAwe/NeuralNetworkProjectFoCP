#pragma once
#include <fstream>
#include <string>
#include <vector>

// Custom getline function
bool my_getline(std::ifstream& file, std::string& line);
std::vector<std::string> get_strings_vector_from_line(const std::string& line);
std::vector<float> get_floats_vector_from_line(const std::string& line);