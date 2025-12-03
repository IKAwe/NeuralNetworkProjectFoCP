#include "data_preprocessor.h"
#include "file_data_operations.h"
#include <iostream>
#include <stdexcept>


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
 * @param data 3-dimensional vector of strings - the dataset to fit, which must include header row
 */
void DataPreprocessor::fit(const std::vector<std::vector<std::string>>& data) {

    if (data.empty()) return;
    
    clear();
    column_order = data[0];
    //Get column names from first row
	//Then check second row for data types - just to be faster
    for (size_t i = 0; i < column_order.size(); i++) {
        ColumnData info;
		info.isNumeric = isNumber(data[1][i]);//(The zero row is header, so check second row)

        if (!info.isNumeric) {
            info.categories.push_back(data[1][i]);
            info.category_to_index[data[1][i]] = 0;
        }

        column_map[data[0][i]] = info;
    }
    // Collect all categories from all rows
	bool is_first_row = true;
    for (const auto& row : data) {
		if (is_first_row) { is_first_row = false; continue; }//Skip first row (header)
        for (size_t i = 0; i < column_order.size(); i++) {
            const std::string& col_name = column_order[i];
            ColumnData& info = column_map[col_name];
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

    update_column_indices();
}

/**
 * @brief Transforms given dataset vector with numeric data and one-hot encodings (the preprocessor must fit the data first) - saves to internal transformed_dataset vector
 * @param data 3-dimensional vector of strings - the dataset to transform
 */
void DataPreprocessor::transform(const std::vector<std::vector<std::string>>& data) {
    if (data.empty()) return;

    transformed_dataset.clear();
    transformed_dataset.reserve(data.size());

    for (const auto& row : data) {
        if (row.size() != column_order.size()) {
            std::cerr << "Row size mismatch with fitted columns";
        }

        std::vector<float> encoded_row;

        for (size_t i = 0; i < column_order.size(); i++) {
            const std::string& col_name = column_order[i];
			const ColumnData& info = column_map.at(col_name);//Dont use [] to avoid creating new keys
            const std::string& value = row[i];

            if (info.isNumeric) {
                // Converting to float
                try {
                    encoded_row.push_back(value.empty() ? 0.0f : stof(value));
                }
                catch (...) {
                    encoded_row.push_back(0.0f);
                }
            }
            else {
                // One-hot encoding
                int cat_number = (int)(info.categories.size());
                int category_index = -1;

                // Find the index of category

                for (int i = 0; i < cat_number; ++i) {
                    if (info.categories[i] == value) {
                        category_index = i;
                        break;
                    }
                }
                // Encode
                for (int j = 0; j < cat_number; ++j) {
                    if (j == category_index) {
                        encoded_row.push_back(1.0f);
                    }
                    else {
                        encoded_row.push_back(0.0f);
                    }
                }
            }
        }
        transformed_dataset.push_back(encoded_row);
    }
}

/**
 * @brief Get and remove a column from the transformed dataset
 * @param columnName 
 * @return 
 */
std::vector<std::vector<float>> DataPreprocessor::extract_column(const std::string& columnName) {
    // Check if column exists
    if (column_map.find(columnName) == column_map.end()) {
        std::cerr << "Column '" << columnName << "' not found";
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
    update_column_indices();

    return extracted_data;
}

// Update column indices after extraction
void DataPreprocessor::update_column_indices() {
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
void DataPreprocessor::fit_transform(const std::vector<std::vector<std::string>>& data) {
    fit(data);
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