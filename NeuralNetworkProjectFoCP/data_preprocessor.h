#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H

#include <vector>
#include <string>
#include <unordered_map>

// Structure to hold information about each column
struct ColumnData {
    bool isNumeric;                           
    std::vector<std::string> categories;      // All unique categories for categorical columns
    std::unordered_map<std::string, int> category_to_index;
	float range[2];                             // Min and Max for numeric columns
};

class DataPreprocessor {
private:
	std::vector<ColumnData> columns;  // Column index -> column_data
    std::unordered_map<std::string, int> column_order;
	std::string extracted_column_name;               
	bool is_fitted = false;                        

public:
    void clear();

    void fit(const std::vector<std::vector<std::string>>& data);

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> transform_and_extract(const std::vector<std::vector<std::string>>& data, const std::string& column_to_extract, bool is_there_header =true);

	void interpret_extracted_column(const std::vector<float>& transformed_data) const;
    void print_state() const;
};

#endif // DATAPREPROCESSOR_H