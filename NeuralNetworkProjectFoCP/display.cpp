#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void print_header(const std::string& text) {
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";  // line of '='
    std::cout << std::setfill(' ') << std::setw(60) << std::left << text << "\n";
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";
	std::cout << std::setfill(' '); // reset fill to space
}



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


void print_feedforward_output(const std::vector<std::vector<float>>& feedforward_input,
    const std::vector<std::vector<float>>& target_output,
    const std::vector<std::vector<float>>& feedforward_output,
    size_t sample_count = 5) {
	print_header(" Feedforward Output ");
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