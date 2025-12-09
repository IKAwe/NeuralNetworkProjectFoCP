#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H

#include <vector>
#include <string>
#include <unordered_map>

// Structure to hold information about each column
struct ColumnData {
    bool isNumeric;                           // Whether column contains numeric data
    std::vector<std::string> categories;      // All unique categories for categorical columns
};

class DataPreprocessor {
private:
	std::vector<ColumnData> columns;  // Column index -> column_data
    std::unordered_map<std::string, int> column_order;
	std::string extracted_column_name;               // Name of the extracted column


public:
    void clear();

    void fit(const std::vector<std::vector<std::string>>& data);

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract);

    // Debug/print current state
    void print_state() const;
};

#endif // DATAPREPROCESSOR_H