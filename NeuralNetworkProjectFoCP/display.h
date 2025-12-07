#pragma once
#ifndef DISPLAY_H
#define DISPLAY_H

#include <string>

// Prints a main header 
void print_header1(const std::string& text);

// Prints a secondary header (less significant)
void print_header2(const std::string& text, int width = 60);

// Prints a separator line (e.g. "-----")
void print_separator(char sep = '-', int width = 60);

#endif // DISPLAY_H