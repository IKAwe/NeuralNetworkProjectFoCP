#include "data_preprocessor.h"
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include "display.h"



void DataPreprocessor::clear() {
    column_order.clear();

	columns.clear();
}



/**
 * @brief Acquire data types(numeric and categrical) and all the categories for all the columns from dataset
 * @param data 3-dimensional vector of strings - the dataset to fit, which must include header row
 */
void DataPreprocessor::fit(const std::vector<std::vector<std::string>>& data) {

    if (data.empty()) return;

    clear();
    //Get column names from first row
    //Then check second row for data types - just to be faster
    for (size_t i = 0; i < data[0].size(); i++) {
        ColumnData info;
        bool is_number = true;

        //Check first data row (second row overall) to determine which col is numeric
        try {
            std::stof(data[1][i]);
        }
        catch (...) {
            is_number = false;
        }
        info.isNumeric = is_number;
        columns.push_back(info);
        column_order[data[0][i]] = i; // Map column name to its index
    }
    // Collect all categories from all rows
    bool is_first_row = true;
    for (const auto& row : data) {
        if (is_first_row) { is_first_row = false; continue; }//Skip first row (header)
        for (int i = 0; i < columns.size(); i++) {
            const auto& col_info = columns[i];
            if (!col_info.isNumeric) {
                bool found = false;
                //Find if category already exists
                for (const auto& category : col_info.categories) {
                    if (category == row[i]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    columns[i].categories.push_back(row[i]);
                }
            }
        }
    }
}

/**
 * @brief Transforms given dataset vector with numeric data and one-hot encodings (the preprocessor must fit the data first) - saves to internal transformed_dataset vector
 * @param data 3-dimensional vector of strings - the dataset to transform
 */
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> DataPreprocessor::transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract) {

	std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> result;
    
    if (data.empty()) return result;

    result.first.reserve(data.size()-1);
	result.second.reserve(data.size() - 1);
	extracted_column_name = column_to_extract;

    bool is_first_row = true;
    for (const auto& row : data) {
        //Maybe cahnge that 
        if (row.size() != columns.size()) {
            std::cerr << "Row size mismatch with fitted columns";
        }
		if (is_first_row) { is_first_row = false; continue; }//Skip first row (header)

        std::vector<float> encoded_row_main;
		std::vector<float> encoded_row_extracted;

		std::vector<float>* current_row = &encoded_row_main;

        for (int i = 0; i < columns.size(); i++) {
            const std::string& value = row[i];
            if (column_order.at(column_to_extract) == i) {
                current_row = &encoded_row_extracted;
            }
            if (columns[i].isNumeric) {
                // Converting to float
                float val_to_be_added = 0.0f;
                try {
                    val_to_be_added = value.empty() ? 0.0f : stof(value);
                }
                catch (...) {
					std::cerr << "Non-numeric value found in numeric column number '" << i << "': " << value << "\n";
                  
                }
				(*current_row).push_back(val_to_be_added);
            }
            else {
                // One-hot encoding
                for (size_t j = 0; j < columns[i].categories.size(); j++) {
                    if (columns[i].categories[j] == value) {
                        (*current_row).push_back(1.0f);
                    }
                    else {
                        (*current_row).push_back(0.0f);
                    }
				}
                
            }
        }
        result.first.push_back(encoded_row_main);
        result.second.push_back(encoded_row_extracted);
    }
	return result;
}

void DataPreprocessor::print_state() const {

    print_header1("DataPreprocessor State");


    std::cout << "Extracted Column: " << extracted_column_name << "\n";
    std::cout << "Total Columns: " << columns.size() << "\n\n";

    std::cout << std::left << std::setw(20) << "Column Name"
        << std::setw(12) << "Type"
        << "Categories\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& name_and_index : column_order) {
        const auto& col = columns[name_and_index.second];
        std::cout << std::left << std::setw(20) << name_and_index.first
            << std::setw(12) << (col.isNumeric ? "Numeric" : "Categorical");

        if (!col.isNumeric && !col.categories.empty()) {
            for (size_t i = 0; i < col.categories.size(); ++i) {
                std::cout << col.categories[i];
                if (i < col.categories.size() - 1) std::cout << ", ";
            }
        }
        std::cout << "\n";
    }

    std::cout << std::string(60, '=') << "\n";

}