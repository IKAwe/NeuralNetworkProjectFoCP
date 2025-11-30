#include "data_preprocessor.h"
#include <stdexcept>

DataPreprocessor::DataPreprocessor() {}

DataPreprocessor::DataPreprocessor(const std::vector<std::string>& names) : column_names(names) {}

void DataPreprocessor::fit(const std::vector<std::vector<std::string>>& data) {
    if (data.empty()) return;

    int num_cols = data[0].size();

    // Initialize column names if not provided
    if (column_names.empty()) {
        column_names.reserve(num_cols);
        for (int i = 0; i < num_cols; i++) {
            column_names.push_back("col_" + std::to_string(i));
        }
    }

    categories.resize(num_cols);
    is_numeric.resize(num_cols, true);
    column_starts.resize(num_cols);
    column_sizes.resize(num_cols);

    // Detect numeric vs categorical columns
    for (const auto& row : data) {
        for (int col = 0; col < num_cols; col++) {
            if (!is_numeric[col]) continue;

            try {
                std::stof(row[col]);
            }
            catch (...) {
                is_numeric[col] = false;
            }
        }
    }

    // Collect categories
    for (const auto& row : data) {
        for (int col = 0; col < num_cols; col++) {
            if (!is_numeric[col]) {
                add_if_new(col, row[col]);
            }
        }
    }

    // Calculate output ranges and build feature mapping
    int current_start = 0;
    for (int col = 0; col < num_cols; col++) {
        column_starts[col] = current_start;
        column_sizes[col] = is_numeric[col] ? 1 : categories[col].size();

        // Build feature mapping
        if (is_numeric[col]) {
            feature_to_col.push_back(col);
            feature_to_cat.push_back(-1); // -1 indicates numeric
            current_start += 1;
        }
        else {
            for (int cat_idx = 0; cat_idx < categories[col].size(); cat_idx++) {
                feature_to_col.push_back(col);
                feature_to_cat.push_back(cat_idx);
            }
            current_start += categories[col].size();
        }
    }

    fitted = true;
}

std::vector<std::vector<float>> DataPreprocessor::encode(const std::vector<std::vector<std::string>>& data) {
    if (!fitted) throw std::runtime_error("Fit first!");

    std::vector<std::vector<float>> result;
    for (const auto& row : data) {
        std::vector<float> encoded_row;
        encoded_row.reserve(feature_to_col.size());

        for (int col = 0; col < row.size(); col++) {
            if (is_numeric[col]) {
                encoded_row.push_back(std::stof(row[col]));
            }
            else {
                int idx = find_index(col, row[col]);
                for (int i = 0; i < categories[col].size(); i++) {
                    encoded_row.push_back(i == idx ? 1.0f : 0.0f);
                }
            }
        }
        result.push_back(encoded_row);
    }
    return result;
}

void DataPreprocessor::get_output_range(int input_col, int& start_idx, int& size) {
    if (!fitted) throw std::runtime_error("Fit first!");
    if (input_col < 0 || input_col >= column_starts.size()) {
        throw std::runtime_error("Invalid column index");
    }
    start_idx = column_starts[input_col];
    size = column_sizes[input_col];
}

void DataPreprocessor::get_feature_info(int output_idx, int& input_col, int& category_idx, std::string& feature_name) {
    if (!fitted) throw std::runtime_error("Fit first!");
    if (output_idx < 0 || output_idx >= feature_to_col.size()) {
        throw std::runtime_error("Invalid output index");
    }

    input_col = feature_to_col[output_idx];
    category_idx = feature_to_cat[output_idx];

    if (category_idx == -1) {
        feature_name = column_names[input_col];
    }
    else {
        feature_name = column_names[input_col] + "_" + categories[input_col][category_idx];
    }
}

int DataPreprocessor::get_output_size() {
    if (!fitted) throw std::runtime_error("Fit first!");
    return feature_to_col.size();
}

std::string DataPreprocessor::get_column_name(int input_col) {
    if (input_col < 0 || input_col >= column_names.size()) {
        throw std::runtime_error("Invalid column index");
    }
    return column_names[input_col];
}

void DataPreprocessor::add_if_new(int col, const std::string& value) {
    for (const auto& cat : categories[col]) {
        if (cat == value) return;
    }
    categories[col].push_back(value);
}

int DataPreprocessor::find_index(int col, const std::string& value) {
    for (int i = 0; i < categories[col].size(); i++) {
        if (categories[col][i] == value) return i;
    }
    throw std::runtime_error("Category not found");
}




std::vector<std::vector<float>> DataPreprocessor::extract_and_remove_column(
    std::vector<std::vector<float>>& encoded_data,
    int input_col_index) {

    if (!fitted) throw std::runtime_error("Fit first!");
    if (input_col_index < 0 || input_col_index >= column_starts.size()) {
        throw std::runtime_error("Invalid column index");
    }

    int start_idx, size;
    get_output_range(input_col_index, start_idx, size);

    std::vector<std::vector<float>> extracted_features;
    extracted_features.reserve(encoded_data.size());

    // Extract the features for this column from each row
    for (auto& row : encoded_data) {
        std::vector<float> extracted_row;
        extracted_row.reserve(size);

        // Copy the features for this column
        for (int i = start_idx; i < start_idx + size; i++) {
            extracted_row.push_back(row[i]);
        }
        extracted_features.push_back(extracted_row);

        // Remove from original encoded data
        row.erase(row.begin() + start_idx, row.begin() + start_idx + size);
    }

    // Update internal indices for remaining columns
    // Shift all column starts that come after the removed column
    for (int col = input_col_index + 1; col < column_starts.size(); col++) {
        column_starts[col] -= size;
    }

    // Remove the column from internal structures
    column_names.erase(column_names.begin() + input_col_index);
    categories.erase(categories.begin() + input_col_index);
    is_numeric.erase(is_numeric.begin() + input_col_index);
    column_starts.erase(column_starts.begin() + input_col_index);
    column_sizes.erase(column_sizes.begin() + input_col_index);

    // Rebuild feature mapping
    feature_to_col.clear();
    feature_to_cat.clear();
    int current_start = 0;
    for (int col = 0; col < column_starts.size(); col++) {
        column_starts[col] = current_start;

        if (is_numeric[col]) {
            feature_to_col.push_back(col);
            feature_to_cat.push_back(-1);
            current_start += 1;
        }
        else {
            for (int cat_idx = 0; cat_idx < categories[col].size(); cat_idx++) {
                feature_to_col.push_back(col);
                feature_to_cat.push_back(cat_idx);
            }
            current_start += categories[col].size();
        }
    }

    return extracted_features;
}