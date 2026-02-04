#include "math_functions.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>



const std::unordered_map<std::string, ActivationFunction> activation_map = {
    {"sigmoid", {sigmoid, sigmoid_derivative}},
    {"relu",    {relu,    relu_derivative}},
    {"tanh",    {mtanh,   mtanh_derivative}}
};


const std::unordered_map<std::string, LossFunction> loss_function_map = {
    {"MSE", {mean_squared_error, mean_squared_error_derivative}},
    {"BCE", {cross_entropy, cross_entropy_derivative}}
};

//================= Activation Functions =================//

//RELU

std::vector<float> relu(const std::vector<float> &x) {
	std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return result;
}

std::vector<float> relu_derivative(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
    return result;
}

//SIGMOID

std::vector<float> sigmoid(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return result;
}

std::vector<float> sigmoid_derivative(const std::vector<float> &x) {
    std::vector<float> sig = sigmoid(x);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sig[i] * (1.0f - sig[i]);
    }
    return result;
}

//TANH

std::vector<float> mtanh(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::tanh(x[i]);
    }
    return result;
}

std::vector<float> mtanh_derivative(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float t = std::tanh(x[i]);
        result[i] = 1.0f - t * t;
    }
    return result;
}



//================= Loss Functions =================//


std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual) {
	std::vector<float> loss(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss[i] = 0.5f * (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
	return loss;
}


std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        gradient[i] = predicted[i] - actual[i];
    }
    return gradient;
}

std::vector<float> cross_entropy(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> loss(predicted.size());
    float epsilon = 1e-7f; // To avoid log(0)
    for (size_t i = 0; i < predicted.size(); ++i) {
        float p = std::max(std::min(predicted[i], 1.0f - epsilon), epsilon);
        loss[i] = -(actual[i] * std::log(p) + (1.0f - actual[i]) * std::log(1.0f - p));
    }
    return loss;
}

std::vector<float> cross_entropy_derivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> gradient(predicted.size());
    float epsilon = 1e-7f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float p = std::max(std::min(predicted[i], 1.0f - epsilon), epsilon);
        gradient[i] = (p - actual[i]) / (p * (1.0f - p));
    }
    return gradient;
}