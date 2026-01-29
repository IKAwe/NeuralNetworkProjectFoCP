#pragma once
#ifndef DISPLAY_H
#define DISPLAY_H

#include <string>
/** @file display.h
 * @brief Header file for display functions.
 */

/**
 * @brief Prints a stylized header to the console.
 * @details The header is surrounded by "=" characters from upper and lower side.
 * @param text The string to be displayed within the header.
 */
void print_header(const std::string& text);
/**
 * @brief Prints a 1D vector of floats in a formatted horizontal array.
 * @details Output format is `[ a b c ... ]` with each value fixed to 2 decimal
 * places and a width of 6 characters for alignment.
 * @param vec The vector of floats to display.
 */
void print_vector(const std::vector<float>& vec);

/**
 * @brief Displays a comparison between inputs, targets, and actual model predictions.
 * @details Useful for visually inspecting model performance on a small sample of data.
 * @param feedforward_input The original feature data.
 * @param target_output The true, target labels.
 * @param feedforward_output The predictions of the NeuralNetwork.
 * @param sample_count The maximum number of samples to print (5 is default).
 */
void print_feedforward_output(const std::vector<std::vector<float>>& feedforward_input,
								const std::vector<std::vector<float>>& target_output,
								const std::vector<std::vector<float>>& feedforward_output,
								const size_t sample_count = 5);
#endif 