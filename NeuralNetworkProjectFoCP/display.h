#pragma once
#ifndef DISPLAY_H
#define DISPLAY_H

#include <string>

// Prints a main header 
void print_header(const std::string& text);

void print_vector(const std::vector<float>& vec);

void print_feedforward_output(const std::vector<std::vector<float>>& feedforward_input,
								const std::vector<std::vector<float>>& target_output,
								const std::vector<std::vector<float>>& feedforward_output,
								const size_t sample_count = 5);
#endif 