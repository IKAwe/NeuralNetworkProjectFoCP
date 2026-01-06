#pragma once


#include <vector>
#include <string>
#include <map>
#include <functional>

std::vector<float> relu(const std::vector<float>& x);
std::vector<float> relu_derivative(const std::vector<float>& x);

std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoid_derivative(const std::vector<float>& x);

std::vector<float> mtanh(const std::vector<float>& x);
std::vector<float> mtanh_derivative(const std::vector<float>& x);

std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual);
std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual);

float mean_squared_error_scalar(const float& predicted, const float& actual);

struct ActivationFunction {
	std::function<std::vector<float>(const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&)> derivative;
};

struct LossFunction {
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> derivative;
};

extern const std::map<std::string, ActivationFunction> activation_map;
extern const std::map<std::string, LossFunction> loss_function_map;



// Utility function to sum elements of a vector
float sum_vector(const std::vector<float>& vec);