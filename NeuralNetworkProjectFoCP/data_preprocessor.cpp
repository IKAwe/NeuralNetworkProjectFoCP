#include "data_preprocessor.h"
#include <iostream>
#include <stdexcept>
//#include <algorithm>


/**
 * @brief Function to check if a string can be converted to a number
 * @param s The string to check
 * @return True if the string is a number, false otherwise
 */
bool DataPreprocessor::isNumber(const std::string& s) {
    if (s.empty()) return true;  // Empty treated as numeric 0

    try {
        std::stof(s);
        return true;
    }
    catch (...) {
        return false;
    }
}
/**
 * @brief Clear all class data
 */
void DataPreprocessor::clear() {
    column_map.clear();
    column_order.clear();
    transformed_dataset.clear();
}



/**
 * @brief Acquire data types(numeric and categrical) and all the categories for all the columns from dataset
 * @param data 3-dimensional vector of strings - the dataset to fit
 * @param column_names vector of strings - the column names
 */
void DataPreprocessor::fit(const std::vector<std::vector<std::string>>& data,const std::vector<std::string>& column_names) {

    if (data.empty()) return;
    if (data[0].size() != column_names.size()) {
        throw std::runtime_error("Column count mismatch");
    }

    clear();
    column_order = column_names;

    // Step 1: Check first row for data types
    for (size_t i = 0; i < column_names.size(); i++) {
        ColumnData info;
        info.isNumeric = isNumber(data[0][i]);

        if (!info.isNumeric) {
            info.categories.push_back(data[0][i]);
            info.category_to_index[data[0][i]] = 0;
        }

        column_map[column_names[i]] = info;
    }

    // Step 2: Collect all categories from all rows
    for (const auto& row : data) {
        for (size_t i = 0; i < column_names.size(); i++) {
            const std::string& colName = column_names[i];
            ColumnData& info = column_map[colName];
            const std::string& value = row[i];

            if (!info.isNumeric) {
                if (info.category_to_index.find(value) == info.category_to_index.end()) {
                    int last_el_index = info.categories.size();
                    info.categories.push_back(value);
                    info.category_to_index[value] = last_el_index;
                }
            }
        }
    }

    updateColumnIndices();
}

/**
 * @brief Transforms given dataset vector with numeric data and one-hot encodings - saves to internal transformed_dataset vector
 * @param data 3-dimensional vector of strings - the dataset to transform
 */
void DataPreprocessor::transform(const std::vector<std::vector<std::string>>& data) {
    if (data.empty()) return;

    transformed_dataset.clear();
    transformed_dataset.reserve(data.size());

    for (const auto& row : data) {
        if (row.size() != column_order.size()) {
            throw std::runtime_error("Row size mismatch with fitted columns");
        }

        std::vector<float> encodedRow;

        for (size_t i = 0; i < column_order.size(); i++) {
            const std::string& colName = column_order[i];
			const ColumnData& info = column_map.at(colName);//Dont use [] to avoid creating new keys
            const std::string& value = row[i];

            if (info.isNumeric) {
                // Convert to float
                try {
                    encodedRow.push_back(value.empty() ? 0.0f : stof(value));
                }
                catch (...) {
                    encodedRow.push_back(0.0f);
                }
            }
            else {
                // One-hot encoding
                int numCats = info.categories.size();
                auto it = info.category_to_index.find(value);

                if (it != info.category_to_index.end()) {
                    int idx = it->second;
                    for (int j = 0; j < numCats; j++) {
                        encodedRow.push_back(j == idx ? 1.0f : 0.0f);
                    }
                }
                else {
                    // Unknown category - all zeros
                    for (int j = 0; j < numCats; j++) {
                        encodedRow.push_back(0.0f);
                    }
                }
            }
        }

        transformed_dataset.push_back(encodedRow);
    }
}

/**
 * @brief Get and remove a column from the transformed dataset
 * @param columnName 
 * @return 
 */
std::vector<std::vector<float>> DataPreprocessor::extractColumn(const std::string& columnName) {
    // Check if column exists
    if (column_map.find(columnName) == column_map.end()) {
        throw std::runtime_error("Column '" + columnName + "' not found");
    }

    // Get column info
    const ColumnData& info = column_map[columnName];
    int startIdx = info.start_index;
    int width = info.total_width;

    // Extract column data
    std::vector<std::vector<float>> extracted_data;
    extracted_data.reserve(transformed_dataset.size());

    for (const auto& row : transformed_dataset) {
        std::vector<float> column_values;
        for (int i = 0; i < width; i++) {
            column_values.push_back(row[startIdx + i]);
        }
        extracted_data.push_back(column_values);
    }

    // Remove column from data structures
    column_order.erase(
        remove(column_order.begin(), column_order.end(), columnName),
        column_order.end()
    );
    column_map.erase(columnName);

    // Remove from transformedData
    for (auto& row : transformed_dataset) {
        row.erase(row.begin() + startIdx, row.begin() + startIdx + width);
    }

    // Update indices for remaining columns
    updateColumnIndices();

    return extracted_data;
}

// Update column indices after extraction
void DataPreprocessor::updateColumnIndices() {
    int currentIndex = 0;

    for (const auto& colName : column_order) {
        ColumnData& info = column_map[colName];
        info.start_index = currentIndex;

        if (info.isNumeric) {
            info.total_width = 1;
            currentIndex += 1;
        }
        else {
            info.total_width = info.categories.size();
            currentIndex += info.total_width;
        }
    }
}





// Fit and transform in one call
void DataPreprocessor::fit_transform(const std::vector<std::vector<std::string>>& data,const std::vector<std::string>& columnNames) {
    fit(data, columnNames);
    transform(data);
}




// Getters
const std::vector<std::vector<float>>& DataPreprocessor::getTransformedData() const {
    return transformed_dataset;
}

const std::unordered_map<std::string, ColumnData>& DataPreprocessor::getColumnInfo() const {
    return column_map;
}

const std::vector<std::string>& DataPreprocessor::getColumnOrder() const {
    return column_order;
}

// Debug print
void DataPreprocessor::printState() const {
    std::cout << "=== DataPreprocessor's State ===\n";
    std::cout << "Columns: ";
    for (const auto& col : column_order) std::cout << col << " ";
    std::cout << "\n\n";

    std::cout << "Column Details:\n";
    for (const auto& col : column_order) {
        const ColumnData& info = column_map.at(col);
        std::cout << "  " << col << ": ";
        if (info.isNumeric) {
            std::cout << "numeric [position " << info.start_index << "]\n";
        }
        else {
            std::cout << "categorical - " << info.categories.size()
                << " categories [starts at " << info.start_index << "]\n";
            for (size_t i = 0; i < info.categories.size(); i++) {
                std::cout << "    [" << i << "] = " << info.categories[i] << "\n";
            }
        }
    }

    if (!transformed_dataset.empty()) {
        std::cout << "\nTransformed Data (" << transformed_dataset.size()
            << " rows, " << transformed_dataset[0].size() << " features):\n";
        for (size_t i = 0; i < std::min(transformed_dataset.size(), size_t(3)); i++) {
            std::cout << "  Row " << i << ": ";
            for (float val : transformed_dataset[i]) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        if (transformed_dataset.size() > 3) {
            std::cout << "  ... and " << (transformed_dataset.size() - 3) << " more rows\n";
        }
    }
}