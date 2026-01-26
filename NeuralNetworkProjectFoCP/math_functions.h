#pragma once
/** @file math_functions.h
 *  @brief Declaration of various mathematical functions and their derivatives.
 */

#include <vector>
#include <string>
#include <map>
#include <functional>

 /**
  * @brief Applies the Rectified Linear Unit (ReLU) activation: \f$ f(x) = \max(0, x) \f$
  * @param x Input vector of weighted sums.
  * @return Vector with ReLU applied element-wise.
  */
std::vector<float> relu(const std::vector<float>& x);
/**
 * @brief Derivative of the ReLU function: \f$ f'(x) = 1 \text{ if } x > 0 \text{ else } 0 \f$
 * @param x Input vector.
 * @return Gradient vector.
 */
std::vector<float> relu_derivative(const std::vector<float>& x);

/**
 * @brief Applies the Sigmoid activation: \f$ f(x) = \frac{1}{1 + e^{-x}} \f$
 * @param x Input vector.
 * @return Vector with sigmoid values in range (0, 1).
 */
std::vector<float> sigmoid(const std::vector<float>& x);
/**
 * @brief Derivative of the Sigmoid function: \f$ f'(x) = f(x) \cdot (1 - f(x)) \f$
 * @param x Input vector.
 * @return Gradient vector.
 */
std::vector<float> sigmoid_derivative(const std::vector<float>& x);

/**
 * @brief Applies the Hyperbolic Tangent (Tanh) activation: \f$ f(x) = \tanh(x) \f$
 * @param x Input vector.
 * @return Vector with tanh values in range (-1, 1).
 */
std::vector<float> mtanh(const std::vector<float>& x);
/**
 * @brief Applies the derivative of the Tanh activation on vector - 1 - tanh^2(x)
 */
std::vector<float> mtanh_derivative(const std::vector<float>& x);


/**
 * @brief Computes the element-wise Squared Error loss: \f$ L = \frac{1}{2}(pred - actual)^2 \f$
 * @param predicted The output predicted by the network.
 * @param actual The ground truth target values.
 * @return A vector of squared differences.
 */
std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual);
/**
 * @brief Computes the derivative of the MSE loss with respect to the prediction.
 * @details \f$ \frac{\partial L}{\partial pred} = pred - actual \f$
 * @param predicted The predicted output vector.
 * @param actual The actual target vector.
 * @return Gradient vector for backpropagation.
 */
std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual);
/**
 * @brief Computes the scalar Squared Error loss for single values.
 * @param predicted The predicted output value.
 * @param actual The actual target value.
 * @return The squared difference.
 */
float mean_squared_error_scalar(const float& predicted, const float& actual);
/**
 * @struct ActivationFunction
 * @brief Wrapper for an activation function and its corresponding derivative. [
 */
struct ActivationFunction {
	std::function<std::vector<float>(const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&)> derivative;
};
/**
 * @struct LossFunction
 * @brief Wrapper for a loss function and its corresponding derivative.
 */
struct LossFunction {
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> derivative;
};
/** @brief Registry mapping string names (e.g., "relu") to ActivationFunction objects.  */
extern const std::map<std::string, ActivationFunction> activation_map;
/** @brief Registry mapping string names (e.g., "MSE") to LossFunction objects.  */
extern const std::map<std::string, LossFunction> loss_function_map;
