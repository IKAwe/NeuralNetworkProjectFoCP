#include "data_preprocessor.h"
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include "display.h"


/**
 * @brief Clears all the member variables of the DataPreprocessor.
 */
void DataPreprocessor::clear() {
    column_order.clear();
    extracted_column_name.clear();
	columns.clear();
	is_fitted = false;
}

/**
 * @brief Fits the DataPreprocessor to the provided dataset.
 * @param data Dataset to fit the preprocessor on
 */
void DataPreprocessor::fit(const std::vector<std::vector<std::string>>& data) {

    if (data.empty()) return;

    //Since we aquire data whether specific column is numeric or not from second row - it must be the same size as first row
    const int col_nb = data[0].size();
    if (col_nb == 0) return;

	// Check if theres at least one data row
    if (data.size() < 2) {
        std::cerr << "Error: Dataset contains only header or insufficient data.\n";
        return;
    }
    clear();
    columns.resize(col_nb);
    //Get column names from first row
    //Then check second row for data types - just to be faster
    for (int i = 0; i < col_nb; i++) {
        ColumnData info;
        bool is_number = true;

        if (data[1].size() != col_nb) { std::cerr << "Error: Invalid row element number\n"; clear();return; }
        //Check first data row (second row overall) to determine which col is numeric
        try {
            std::stof(data[1][i]);
        }
        catch (...) {
            is_number = false;
        }
        info.isNumeric = is_number;
		columns[i] = info; // Have resized already
        column_order[data[0][i]] = i; // Map column name to its index
    }
    // Collect all categories from all rows
    bool is_first_row = true;
    
    for (const auto& row : data) {
        if (is_first_row) { is_first_row = false; continue; }//Skip first row (header)
        if (row.size() != col_nb) { std::cerr << "Error: Invalid row element number\n"; clear(); return; }
        for (int i = 0; i < col_nb; i++) {
			std::string value = row[i];
            auto& col_data = columns[i];

            if (!col_data.isNumeric) {
                // Check if the category is already known
                if (col_data.category_to_index.find(value) == col_data.category_to_index.end()) {
                    // 1. Get the new index (which is the current size of the vector)
                    int new_index = col_data.categories.size();

                    // 2. Add to the vector (defines the output order)
                    col_data.categories.push_back(value);

                    // 3. Add to the map (defines the fast lookup)
                    col_data.category_to_index[value] = new_index;
                }
            }
        }
    }
	is_fitted = true;
}

/**
 * @brief Transforms and extracts a specific column from the input data.
 * @param data Data to be transformed and from which a column is to be extracted
 * @param column_to_extract Name of the column to extract
 * @return Pair where the first element is the transformed data and the second element is the extracted column data
 */
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> DataPreprocessor::transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract) {

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> result = { {}, {}};
    
    if (data.empty()) return result;

	// Check if fit has been called - whether columns have been defined
    if (!is_fitted) {
        std::cerr << "Error: DataPreprocessor wasn't fit onto any data yet\n";
        return result;
    }
    
	// Check if the column to extract exists
    auto it_extract = column_order.find(column_to_extract);
    if (it_extract == column_order.end()) {
        std::cerr << "Error: Column to extract '" << column_to_extract << "' not found in fitted data.\n";
        return result;
    }
    extracted_column_name = column_to_extract;
    const int extract_index = it_extract->second;

    result.first.reserve(data.size() - 1);
    result.second.reserve(data.size() - 1);
    bool is_first_row = true;

    for (const auto& row : data) {
        //Maybe cahnge that 
        if (row.size() != columns.size()) {
            std::cerr << "Error: Row size mismatch with fitted columns";
			return result;
        }
		if (is_first_row) { is_first_row = false; continue; }//Skip first row (header)

        std::vector<float> encoded_row_main;
		std::vector<float> encoded_row_extracted;

		std::vector<float>* current_row = &encoded_row_main;

        for (int i = 0; i < columns.size(); i++) {
            const std::string& value = row[i];
            if (i == extract_index) {
                current_row = &encoded_row_extracted;
            }
            else {
				current_row = &encoded_row_main;
            }
            if (columns[i].isNumeric) {
                // Converting to float
                float val_to_be_added = 0.0f;
                try {
                    val_to_be_added = value.empty() ? 0.0f : stof(value);
                }
                catch (...) {
					std::cerr << "Error: Non-numeric value found in numeric column number '" << i << "': " << value << "\n";
                }
				(*current_row).push_back(val_to_be_added);
            }
            else {
                // One-hot encoding
                const auto& col_info = columns[i];

                // Find the index of the current value (O(1) average)
                auto map_it = col_info.category_to_index.find(value);

                // Determine the target index, or -1 if the category is unseen
                int target_index = -1;
                if (map_it != col_info.category_to_index.end()) {
                    target_index = map_it->second;
                }
                // Handle unseen categories - all zeros

                size_t num_categories = col_info.categories.size();
                for (size_t j = 0; j < num_categories; j++) {
                    // Push 1.0f only if the current index j matches the target_index
                    float encoded_val = (j == (size_t)target_index) ? 1.0f : 0.0f;
                    (*current_row).push_back(encoded_val);
                }
                
            }
        }
        result.first.push_back(encoded_row_main);
        result.second.push_back(encoded_row_extracted);
    }
	return result;
}

/**
 * @brief Prints the current state of the DataPreprocessor
 */
void DataPreprocessor::print_state() const {
    if (!is_fitted) {
        std::cout << "DataPreprocessor hasn't been fitted yet.\n";
        return;
	}
    print_header1("DataPreprocessor State");

    std::cout << "Extracted Column: " << extracted_column_name << "\n";
    std::cout << "Total Columns: " << columns.size() << "\n\n";

    // Table header
    std::cout << std::left
        << std::setw(20) << "Column Name"
        << " | "
        << std::setw(12) << "Type"
        << " | Categories\n";

    // Separator
    std::cout << std::setfill('-') << std::setw(60) << "" << std::setfill(' ') << "\n";

    for (size_t i = 0; i < columns.size(); ++i) {

        // Find name for this index (Slow)
        std::string col_name = "Unknown";
        for (const auto& pair : column_order) {
            if (pair.second == i) { col_name = pair.first; break; }
        }

        const auto& col = columns[i];

        std::cout << std::left
            << std::setw(20) << col_name
            << " | "
            << std::setw(12) << (col.isNumeric ? "Numeric" : "Categorical")
            << " | ";

        if (!col.isNumeric && !col.categories.empty()) {
            for (size_t i = 0; i < col.categories.size(); ++i) {
                std::cout << col.categories[i];
                if (i < col.categories.size() - 1) std::cout << ", ";
            }
        }

        std::cout << "\n";
    }

	print_separator('-', 60);
}

