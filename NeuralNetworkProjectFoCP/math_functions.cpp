#include "math_functions.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <map>


/**
 * @brief Registry of available activation functions and their derivatives.
 * @details Maps a string name (e.g., "sigmoid") to an ActivationFunction struct
 * containing the primary function and its derivative.
 */
const std::map<std::string, ActivationFunction> activation_map = {
    {"sigmoid", {sigmoid, sigmoid_derivative}},
    {"relu",    {relu,    relu_derivative}},
    {"tanh",    {mtanh,   mtanh_derivative}}
};

/**
 * @brief Registry of available loss functions and their derivatives.
 * @details Maps a string name ("MSE" or "BCE") to a LossFunction struct
 * containing the primary function and its derivative.
 */
const std::map<std::string, LossFunction> loss_function_map = {
    {"MSE", {mean_squared_error, mean_squared_error_derivative}},
    {"BCE", {cross_entropy, cross_entropy_derivative}}
};

//================= Activation Functions =================//

//RELU
/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation: \f$ f(x) = \max(0, x) \f$
 * @param x Input vector of weighted sums.
 * @return Vector with ReLU applied element-wise.
 */
std::vector<float> relu(const std::vector<float> &x) {
	std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return result;
}
/**
 * @brief Derivative of the ReLU function: \f$ f'(x) = 1 \text{ if } x > 0 \text{ else } 0 \f$
 * @param x Input vector.
 * @return Gradient vector.
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
 * @brief Applies the Sigmoid activation: \f$ f(x) = \frac{1}{1 + e^{-x}} \f$
 * @param x Input vector.
 * @return Vector with sigmoid values in range (0, 1).
 */
std::vector<float> sigmoid(const std::vector<float> &x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return result;
}
/**
 * @brief Derivative of the Sigmoid function: \f$ f'(x) = f(x) \cdot (1 - f(x)) \f$
 * @param x Input vector.
 * @return Gradient vector.
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
 * @brief Applies the Hyperbolic Tangent (Tanh) activation: \f$ f(x) = \tanh(x) \f$
 * @param x Input vector.
 * @return Vector with tanh values in range (-1, 1).
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
 * @brief Computes the element-wise Squared Error loss: \f$ L = \frac{1}{2}(pred - actual)^2 \f$
 * @param predicted The output predicted by the network.
 * @param actual The ground truth target values.
 * @return A vector of squared differences.
 */
std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual) {
	std::vector<float> loss(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss[i] = 0.5f * (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
	return loss;
}

/**
 * @brief Computes the derivative of the MSE loss with respect to the prediction.
 * @details \f$ \frac{\partial L}{\partial pred} = pred - actual \f$
 * @param predicted The predicted output vector.
 * @param actual The actual target vector.
 * @return Gradient vector for backpropagation.
 */
std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        gradient[i] = predicted[i] - actual[i];
    }
    return gradient;
}
/**
 * @brief Binary Cross Entropy Loss: L = -[y*log(p) + (1-y)*log(1-p)]
 * @param predicted The output of the sigmoid layer (must be in range 0-1).
 * @param actual The ground truth labels (0 or 1).
 * @return A vector of cross-entropy losses.
 */
std::vector<float> cross_entropy(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> loss(predicted.size());
    float epsilon = 1e-7f; // To avoid log(0)
    for (size_t i = 0; i < predicted.size(); ++i) {
        float p = std::max(std::min(predicted[i], 1.0f - epsilon), epsilon);
        loss[i] = -(actual[i] * std::log(p) + (1.0f - actual[i]) * std::log(1.0f - p));
    }
    return loss;
}
/**
 * @brief Derivative of Binary Cross Entropy Loss with respect to predictions.
 * @param predicted The output of the sigmoid layer (must be in range 0-1).
 * @param actual The ground truth labels (0 or 1).
 * @return Gradient vector for backpropagation.
 */
std::vector<float> cross_entropy_derivative(const std::vector<float>& predicted, const std::vector<float>& actual) {
    std::vector<float> gradient(predicted.size());
    float epsilon = 1e-7f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float p = std::max(std::min(predicted[i], 1.0f - epsilon), epsilon);
        gradient[i] = (p - actual[i]) / (p * (1.0f - p));
    }
    return gradient;
}