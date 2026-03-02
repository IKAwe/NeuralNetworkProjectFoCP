#pragma once
/** @file math_functions.h
 *  @brief Declaration of various mathematical functions and their derivatives.
 */

#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <functional>

 /**
  * @brief Applies the Rectified Linear Unit (ReLU) activation: \f$ f(x) = \max(0, x) \f$
  * @param x Input vector.
  * @return Vector with ReLU applied.
  */
std::vector<float> relu(const std::vector<float>& x);
/**
 * @brief Derivative of the ReLU function: \f$ f'(x) = 1 \text{ if } x > 0 \text{ else } 0 \f$
 * @param x Input vector.
 * @return Gradient vector.
 */
std::vector<float> relu_derivative(const std::vector<float>& x);

/**
 * @brief Applies the sigmoid activation: \f$ f(x) = \frac{1}{1 + e^{-x}} \f$
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
 * @brief Applies the hyperbolic tangent (Tanh) activation: \f$ f(x) = \tanh(x) \f$
 * @param x Input vector.
 * @return Vector with tanh values in range (-1, 1).
 */
std::vector<float> mtanh(const std::vector<float>& x);
/**
 * @brief Applies the derivative of the Tanh activation on vector - 1 - tanh^2(x)
 * @param x Input vector.
 * @return Gradient vector.
 */
std::vector<float> mtanh_derivative(const std::vector<float>& x);


/**
 * @brief Computes the Squared Error loss: \f$ L = \frac{1}{2}(pred - actual)^2 \f$
 * @param predicted The output predicted by the network.
 * @param actual The true target values.
 * @return A vector of squared differences.
 */
std::vector<float> mean_squared_error(const std::vector<float>& predicted, const std::vector<float>& actual);
/**
 * @brief Computes the derivative of the MSE loss with respect to the prediction.
 * @details \f$ \frac{\partial L}{\partial pred} = pred - actual \f$
 * @param predicted The predicted output vector.
 * @param actual The actual target vector.
 * @return Gradient vector.
 */
std::vector<float> mean_squared_error_derivative(const std::vector<float>& predicted, const std::vector<float>& actual);

/**
 * @brief Binary Cross Entropy Loss: L = -[y*log(p) + (1-y)*log(1-p)]
 * @param predicted The vector of predicted probabilities (must be in range 0-1).
 * @param actual The true target labels (0 or 1).
 * @return A vector of cross-entropy losses.
 */
std::vector<float> cross_entropy(const std::vector<float>& predicted, const std::vector<float>& actual);
/**
 * @brief Derivative of Binary Cross Entropy Loss with respect to predictions.
 * @param predicted The vector of predicted probabilities (must be in range 0-1).
 * @param actual The true target labels (0 or 1).
 * @return Gradient vector.
 */
std::vector<float> cross_entropy_derivative(const std::vector<float>& predicted, const std::vector<float>& actual);
/**
 * @struct ActivationFunction
 * @brief Wrapper for an activation function and its derivative. [
 */
struct ActivationFunction {
	std::function<std::vector<float>(const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&)> derivative;
};
/**
 * @struct LossFunction
 * 
 * @brief Wrapper for a loss function and its derivative.
 */
struct LossFunction {
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> func;
	std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)> derivative;
};
/** @brief Maps string names (e.g., "relu") to ActivationFunction objects.  */
extern const std::unordered_map<std::string, ActivationFunction> activation_map;
/** @brief Maps string names (e.g., "MSE") to LossFunction objects.  */
extern const std::unordered_map<std::string, LossFunction> loss_function_map;
