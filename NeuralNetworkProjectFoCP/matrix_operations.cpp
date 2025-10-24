#include "matrix_operations.h";



std::vector<std::vector <double>>matrixMultiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
	if (A.empty() || B.empty() || A[0].size() != B.size()) {
		//throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
	}
	size_t rowsA = A.size();
	size_t colsA = A[0].size();
	size_t colsB = B[0].size();
	std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
	for (size_t i = 0; i < rowsA; ++i) {
		for (size_t j = 0; j < colsB; ++j) {
			for (size_t k = 0; k < colsA; ++k) {
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return result;
}