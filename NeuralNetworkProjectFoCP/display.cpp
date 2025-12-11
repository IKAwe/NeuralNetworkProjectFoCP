#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void print_header1(const std::string& text) {
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";  // line of '='
    std::cout << std::setfill(' ') << std::setw(60) << std::left << text << "\n";
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";
	std::cout << std::setfill(' '); // reset fill to space
}

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

void print_separator(char sep, int width) {
    std::cout << std::setfill(sep) << std::setw(width) << "" << "\n";
    std::cout << std::setfill(' '); 
	std::cout << std::endl;
}


void print_vector(const std::vector<float>& vec) {
    std::cout << "[ ";

    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << vec[i];

        /*if (i < vec.size() - 1) {
            std::cout << ", ";
        }*/
    }

    std::cout << " ]\n";
}
