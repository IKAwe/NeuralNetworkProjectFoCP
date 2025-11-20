#pragma once
#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <vector>
#include <string>

std::vector<std::vector<std::string>> parseCSV(const std::string& filename);

#endif