#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
/**
 * @brief Print a header with '=' lines above and below the text.
 * @param text The header text to print.
 */
void print_header1(const std::string& text) {
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";  // line of '='
    std::cout << std::setfill(' ') << std::setw(60) << std::left << text << "\n";
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";
	std::cout << std::setfill(' '); // reset fill to space
}

/**
 * @brief Print a header with '=' filling to the sides of the text.
 * @param text The header text to print.
 * @param width The total width of the header.
 */
void print_header2(const std::string& text, int width) {
    int text_len = (int)(text.size());
    int total_fill = width - text_len;
    if (total_fill < 0) total_fill = 0;

    int left_fill = total_fill / 2;
    int right_fill = total_fill - left_fill;

    std::cout << std::setfill('=')
        << std::setw(left_fill) << "" 
        << std::setfill(' ') << text 
        << std::setfill('=')
        << std::setw(right_fill) << ""
        << std::setfill(' ') << "\n";  
}

/**
 * @brief Print a separator line made of a specific character.
 * @param sep The character to use for the separator line.
 * @param width The width of the separator line.
 */
void print_separator(char sep, int width) {
    std::cout << std::setfill(sep) << std::setw(width) << "" << "\n";
    std::cout << std::setfill(' '); 
	std::cout << std::endl;
}

/**
 * @brief Print the contents of a vector of floats ([] format).
 * @param vec The vector to print.
 */
void print_vector(const std::vector<float>& vec) {
    if (vec.empty()) {
        std::cout << "[ ]\n";
        return;
	}
    std::cout << "[ ";

    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << vec[i];

        /*if (i < vec.size() - 1) {
            std::cout << ", ";
        }*/
    }

    std::cout << " ]\n";
}

/**
 * @brief Print the feedforward output along with input and target output for a few samples.
 * @param feedforward_input The input data used for feedforward.
 * @param target_output The target output data.
 * @param feedforward_output The output produced by the feedforward process.
 */
void print_feedforward_output(const std::vector<std::vector<float>>& feedforward_input,
    const std::vector<std::vector<float>>& target_output,
    const std::vector<std::vector<float>>& feedforward_output,
    size_t sample_count = 5) {
	print_header1(" Feedforward Output ");
    size_t samples_nb = std::min(feedforward_input.size(), sample_count);
    for (size_t i = 0; i < samples_nb; ++i) {
        std::cout << "Sample " << i + 1 << ":\n";
        std::cout << "  Input:    ";
        print_vector(feedforward_input[i]);
        std::cout << "  Target:   ";
        print_vector(target_output[i]);
        std::cout << "  Feedforward prediction:   ";
        print_vector(feedforward_output[i]);
        std::cout << "\n";
	}
}