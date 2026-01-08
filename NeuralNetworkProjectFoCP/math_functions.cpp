#include "math_functions.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <map>



const std::map<std::string, ActivationFunction> activation_map = {
    {"sigmoid", {sigmoid, sigmoid_derivative}},
    {"relu",    {relu,    relu_derivative}},
    {"tanh",    {mtanh,   mtanh_derivative}}
};

const std::map<std::string, LossFunction> loss_function_map = {
    {"MSE", {mean_squared_error, mean_squared_error_derivative}}
};

//================= Activation Functions =================//

//RELU
/**
 * @brief Applies the ReLU activation on vector - max(0, x)
 * @param x The input vector.
 * @return The activated vector.
 */
std::vector<float> relu(const std::vector<float> &x) {
	std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return result;
}
/**
 * @brief Applies the derivative of the ReLU activation on vector - (1) if x > 0 else 0
 * @param x The input vector.
 * @return The derivative vector.
 */
std::vector<float> relu_derivative(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
    return result;
}

//SIGMOID
/**
 * @brief Applies the Sigmoid activation on vector - 1 / (1 + exp(-x))
 */
std::vector<float> sigmoid(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return result;
}
/**
 * @brief Applies the derivative of the Sigmoid activation on vector - sigmoid(x) * (1 - sigmoid(x))
 */
std::vector<float> sigmoid_derivative(const std::vector<float> &x) {
    std::vector<float> sig = sigmoid(x);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sig[i] * (1.0f - sig[i]);
    }
    return result;
}

//TANH
/**
 * @brief Applies the Tanh activation on vector - tanh(x)
 */
std::vector<float> mtanh(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::tanh(x[i]);
    }
    return result;
}
/**
 * @brief Applies the derivative of the Tanh activation on vector - 1 - tanh^2(x)
 */
std::vector<float> mtanh_derivative(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float t = std::tanh(x[i]);
        result[i] = 1.0f - t * t;
    }
    return result;
}



//================= Loss Functions =================//

/**
 * @brief Computes the Mean Squared Error (MSE) loss between predicted and actual vectors.
 * @param predicted The predicted output vector.
 * @param actual The actual target vector.
 * @return The MSE loss vector.
 */
std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual) {
	std::vector<float> loss(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss[i] = 0.5f * (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
	return loss;
}

float mean_squared_error_scalar(const float& predicted, const float& actual) {
	return 0.5f * (predicted - actual) * (predicted - actual);
}

std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        gradient[i] = predicted[i] - actual[i];
    }
    return gradient;
}


//========== Vector Utilities ==========//

/**
 * @brief Sums all elements in a vector.
 * @param vec The input vector.
 * @return The sum of the vector elements.
 */
float sum_vector(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (const auto& val : vec) {
        sum += val;
    }
    return sum;
}

