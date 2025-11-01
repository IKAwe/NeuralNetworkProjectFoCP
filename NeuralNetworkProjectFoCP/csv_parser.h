#pragma once
#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <vector>
#include <string>

std::vector<std::vector<float>> parseCSV(const std::string& filename);

#endif