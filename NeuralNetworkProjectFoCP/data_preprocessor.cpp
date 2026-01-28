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
 * @brief Fits the preprocessor by determining column types and calculating normalization ranges.
 * @details Iterates through the dataset to identify numeric vs. categorical columns.
 * For numeric columns, it finds min/max for normalization. For categorical, it maps unique strings to indices.
 * @param data Dataset (2D string vector). The first row **must** be the header.
 * @note This function populates internal maps and clears any previous state.
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
        float value;

        if (data[1].size() != col_nb) { std::cerr << "Error: Invalid row element number\n"; clear();return; }
        //Check first data row (second row overall) to determine which col is numeric
        try {
            value = std::stof(data[1][i]);
        }
        catch (...) {
            is_number = false;
        }
		//Initialize range for numeric columns
        if (is_number) {
            info.range[0] = value; // min
            info.range[1] = value; // max
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
            else {
                // Update min and max for numeric columns - can be optimised
                float num_value = 0.0f;
                try {
                    num_value = std::stof(value);
                    if (num_value < col_data.range[0]) col_data.range[0] = num_value;
                    if (num_value > col_data.range[1]) col_data.range[1] = num_value;
                }
                catch (...) {
                    // Ignore non-numeric values in numeric columns
				}
            }
        }
    }
	is_fitted = true;
}

/**
 * @brief Transforms the dataset using One-Hot Encoding and Min-Max Normalization.
 * @details Separates the dataset into a feature set and a specific target column.
 * @param data The raw string data to transform.
 * @param column_to_extract The name of the column to be used as the target (Y).
 * @param is_there_header Flag indicating if the input 'data' includes a header row.
 * @return A pair of 2D float vectors:
 * - **first**: Transformed feature matrix (all columns except the extracted one).
 * - **second**: Transformed target matrix (the extracted column).
 * @pre The preprocessor must have been successfully fitted using fit().
 */
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> DataPreprocessor::transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract, bool is_there_header) {

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
	// Store the name of the extracted column in member variable
    extracted_column_name = column_to_extract;
    const int extract_index = it_extract->second;

	//preallocate space depending on whether there's a header row
    size_t expected_rows = is_there_header ? (data.size() - 1) : data.size();
    result.first.reserve(expected_rows);
    result.second.reserve(expected_rows);

	// Skip header row
    bool skip_row = is_there_header;
    for (const auto& row : data) {
        if (row.size() != columns.size()) {
            std::cerr << "Error: Row size mismatch with fitted columns";
			// Skip invalid row
			continue;
        }
		if (skip_row) { skip_row = false; continue; }//Skip first row (header)

        std::vector<float> encoded_row_main;
		std::vector<float> encoded_row_extracted;

		// Process each column in the row
        for (int i = 0; i < columns.size(); i++) {
            const std::string& value = row[i];
			const ColumnData& current_column = columns[i];
			// Decide which row to populate using pointer
            std::vector<float>* target_vec = (i == extract_index) ? &encoded_row_extracted : &encoded_row_main;

			// Handle numeric columns
            if (columns[i].isNumeric) {
                // Try converting to float
                float val_to_be_added = 0.0f;
                try {
                    val_to_be_added = value.empty() ? 0.0f : stof(value);
                }
                catch (...) {
					std::cerr << "Error: Non-numeric value found in numeric column number '" << i << "': " << value << "\n";
                }
				// Normalization
				float range_diff = current_column.range[1] - current_column.range[0];
                if (range_diff != 0) {
					(*target_vec).push_back((val_to_be_added-current_column.range[0])/(range_diff));
                }
                else {
					(*target_vec).push_back(0.0f); // If all values are the same push 0.0
                }
            }
			// Handle categorical columns
            else {
                // One-hot encoding
                const auto& col_info = columns[i];
                auto map_it = col_info.category_to_index.find(value);
                int target_index = (map_it != col_info.category_to_index.end()) ? map_it->second : -1;

                size_t num_categories = col_info.categories.size();

                //Capture the size before  growing
                size_t start_idx = target_vec->size();

                // insert all zeros
                target_vec->insert(target_vec->end(), num_categories, 0.0f);

                // Flip the 1 using simple indexing
                if (target_index != -1) {
                    (*target_vec)[start_idx + target_index] = 1.0f;
                }
                
            }
        }
		//Move instead of copy
        result.first.push_back(std::move(encoded_row_main));
        result.second.push_back(std::move(encoded_row_extracted));
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
    print_header("DataPreprocessor State");

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

}


/**
 * @brief Decodes processed float data back into human-readable format.
 * @details Reverses normalization for numeric columns and picks the highest probability
 * category (Argmax) for one-hot encoded categorical columns.
 * @param transformed_data A 1D vector representing a single sample of the extracted column.
 * @pre transform_and_extract() must have been called at least once to set the active column name.
 */
void DataPreprocessor::interpret_extracted_column(const std::vector<float>& transformed_data) const {
    if (!is_fitted) {
        std::cerr << "Error: DataPreprocessor wasn't fit onto any data yet\n";
        return;
    }
    if (extracted_column_name.empty()) {
        std::cerr << "Error: Have not run transform_and_extract yet\n";
        return;
    }
    auto& col_data = columns[column_order.at(extracted_column_name)];

    if (col_data.isNumeric) {
        //the transformed_data vector is like {1}
        std::cout << "Interpreting extracted numeric column '" << extracted_column_name << "':\n";
        float actual_value = col_data.range[0] + (col_data.range[1] - col_data.range[0]) * transformed_data[0];// It was normalized during transform
        std::cout << "  Sample value: " << transformed_data[0] << " -> Actual value (de-normalised): " << actual_value << "\n";
    }
    else {
        std::cout << "Interpreting extracted categorical column '" << extracted_column_name << "':\n";
        std::cout << "  Sample one-hot encoded value: ";
        // Find the index of the max value in transformed_data
        int max_index = 0; //the index of the category with highest probability
        float max_value = transformed_data[0];
        for (size_t i = 0; i < transformed_data.size(); ++i) {
            if (transformed_data[i] > max_value) {
                max_value = transformed_data[i];
                max_index = i;
            }
        }
        print_vector(transformed_data);
        std::cout << " -> Actual category: " << col_data.categories[max_index] << "\n";
    }
}
