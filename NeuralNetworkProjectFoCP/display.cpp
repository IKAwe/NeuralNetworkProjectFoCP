#include <iostream>
#include <iomanip>
#include <string>

void print_header1(const std::string& text) {
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";  // line of '='
    std::cout << std::setfill(' ') << std::setw(60) << std::left << text << "\n";
    std::cout << std::setfill('=') << std::setw(60) << "" << "\n";
}

void print_header2(const std::string& text, int width) {
    int text_len = (int)(text.size());
    int total_fill = width - text_len;
    if (total_fill < 0) total_fill = 0;

    int left_fill = total_fill / 2;
    int right_fill = total_fill - left_fill;

    std::cout << std::setfill('=')
        << std::setw(left_fill) << ""   // left padding
        << std::setfill(' ') << text   // the actual header text
        << std::setfill('=')
        << std::setw(right_fill) << ""  // right padding
        << std::setfill(' ') << "\n";  // reset fill to space
}

void print_separator(char sep, int width) {
    std::cout << std::setfill(sep) << std::setw(width) << "" << "\n";
    std::cout << std::setfill(' '); // reset to space
	std::cout << std::endl;
}