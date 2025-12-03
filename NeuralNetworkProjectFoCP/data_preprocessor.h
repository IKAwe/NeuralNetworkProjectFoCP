#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H

#include <vector>
#include <string>
#include <unordered_map>

// Structure to hold information about each column
struct ColumnData {
    bool isNumeric;                           // Whether column contains numeric data
    std::vector<std::string> categories;      // All unique categories for categorical columns
    std::unordered_map<std::string, int> category_to_index;  // Quick lookup: category -> index
    int start_index;                           // Starting position in transformed data
    int total_width;                           // 1 for numeric, categories.size() for categorical

    ColumnData() : isNumeric(true), start_index(0), total_width(1) {}
};

class DataPreprocessor {
private:
	std::unordered_map<std::string, ColumnData> column_map;  // Column name -> column_data
    std::vector<std::string> column_order;                   // Preserve original column order
    std::vector<std::vector<float>> transformed_dataset;        // Main transformed data (public accessible)

    // Helper methods
    bool isNumber(const std::string& s);
    void update_column_indices();  // Update start indices after column extraction

public:
    void clear();

    void fit(const std::vector<std::vector<std::string>>& data);

    void transform(const std::vector<std::vector<std::string>>& data);

    std::vector<std::vector<float>> extract_column(const std::string& columnName);

    void fit_transform(const std::vector<std::vector<std::string>>& data);

    // Getters for public access
    const std::vector<std::vector<float>>& getTransformedData() const;
    const std::unordered_map<std::string, ColumnData>& getColumnInfo() const;
    const std::vector<std::string>& getColumnOrder() const;

    // Debug/print current state
    void printState() const;
};

#endif // DATAPREPROCESSOR_H