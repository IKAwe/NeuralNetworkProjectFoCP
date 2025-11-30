#ifndef COLUMN_TRACKING_PREPROCESSOR_H
#define COLUMN_TRACKING_PREPROCESSOR_H

#include <vector>
#include <string>

class DataPreprocessor {
private:
    std::vector<std::vector<std::string>> categories;
    std::vector<bool> is_numeric;
    std::vector<std::string> column_names;
    std::vector<int> column_starts; // starting index for each input column
    std::vector<int> column_sizes;  // size for each input column
    std::vector<int> feature_to_col; // output_idx -> input_col_idx
    std::vector<int> feature_to_cat; // output_idx -> category_idx (-1 for numeric)
    bool fitted = false;

    void add_if_new(int col, const std::string& value);
    int find_index(int col, const std::string& value);

public:
    // Constructor with optional column names
    DataPreprocessor();
    DataPreprocessor(const std::vector<std::string>& names);

    // Fit the preprocessor to the data
    void fit(const std::vector<std::vector<std::string>>& data);

    // Encode data to numerical format
    std::vector<std::vector<float>> encode(const std::vector<std::vector<std::string>>& data);

    // Get output range for a specific input column
    void get_output_range(int input_col, int& start_idx, int& size);

    // Get information about a specific output feature
    void get_feature_info(int output_idx, int& input_col, int& category_idx, std::string& feature_name);

    // Get total output size
    int get_output_size();

    // Get column name
    std::string get_column_name(int input_col);

    // Check if fitted
    bool is_fitted() const { return fitted; }

    // Get number of input columns
    int get_input_columns() const { return column_names.size(); }
    // Extract and remove column - returns the extracted column and updates indices
    // Extract and remove column from encoded data - returns the extracted features
    std::vector<std::vector<float>> extract_and_remove_column(
        std::vector<std::vector<float>>& encoded_data,
        int input_col_index);
};

#endif // COLUMN_TRACKING_PREPROCESSOR_H