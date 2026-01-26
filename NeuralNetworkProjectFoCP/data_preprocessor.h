#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H
/** @file data_preprocessor.h
 * @brief Logic for dataset cleaning, normalization, and one-hot encoding.
 */
#include <vector>
#include <string>
#include <unordered_map>

 /**
  * @struct ColumnData
  * @brief Metadata for an individual dataset column.
  */
struct ColumnData {
    bool isNumeric;                           
    std::vector<std::string> categories;      // All unique categories for categorical columns
    std::unordered_map<std::string, int> category_to_index;
	float range[2];                             // Min and Max for numeric columns
};
/**
 * @class DataPreprocessor
 * @brief Handles the conversion of raw CSV string data into normalized float matrices.
 */
class DataPreprocessor {
private:
	std::vector<ColumnData> columns;  // Column index -> column_data
    std::unordered_map<std::string, int> column_order;
	std::string extracted_column_name;               
	bool is_fitted = false;                        

public:
    /**
     * @brief Clears all the member variables of the DataPreprocessor.
     */
    void clear();
    /**
     * @brief Fits the preprocessor by determining column types and calculating normalization ranges.
     * @details Iterates through the dataset to identify numeric vs. categorical columns.
     * For numeric columns, it finds min/max for normalization. For categorical, it maps unique strings to indices.
     * @param data Dataset (2D string vector). The first row **must** be the header.
     * @note This function populates internal maps and clears any previous state.
     */
    void fit(const std::vector<std::vector<std::string>>& data);
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
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract, bool is_there_header =true);
    /**
     * @brief Decodes processed float data back into human-readable format.
     * @details Reverses normalization for numeric columns and picks the highest probability
     * category (Argmax) for one-hot encoded categorical columns.
     * @param transformed_data A 1D vector representing a single sample of the extracted column.
     * @pre transform_and_extract() must have been called at least once to set the active column name.
     */
	void interpret_extracted_column(const std::vector<float>& transformed_data) const;
    /**
     * @brief Prints the current state of the DataPreprocessor
     */
    void print_state() const;
};

#endif // DATAPREPROCESSOR_H