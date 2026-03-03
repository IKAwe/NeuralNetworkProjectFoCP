# Modular C++ Neural Network

### **Overview**
This project is a high-performance, modular implementation of a feedforward neural network built in **C++**. Designed for both classification and regression tasks, it provides a pipeline for handling raw CSV data, automated preprocessing, and model state persistence.

Latest Updates

The latest, optimized version of the network is located in the Documentation_ready branch. This is where I am currently focusing on performance tuning for large datasets.

---

## **Core Engineering Features**

*  **Automated data pipeline**: The `DataPreprocessor` class automatically detects column types.  It applies **Min-Max normalization** to numeric features and **One-Hot encoding** to categorical labels to prepare data for the network.
* **High-performance computing**: Optimized for speed using **OpenMP** to parallelize mathematical operations across multiple cores, reducing feedforward and training time for large datasets.
*  **Flexible architecture**: Fully configurable network depth and layer width.  It supports multiple activation functions (**Sigmoid**, **ReLU**, **Tanh**) and loss functions (**MSE**, **BCE**) with native derivative implementations for backpropagation.
*  **Model persistence**: Save or load trained models— weights, biases, and activations configurations to txt files.
* **Modularity**: Decoupled design between the math, data handling, and CLI parser ensures extensibility for future research and development.

---

## **Command Line Interface (CLI)**

 The application can be controlled via set of arguments parsed into a `Configuration` struct.

### **1. Model configuration**
*  `--initialization <name>`: Weight initialization method—choose **Xavier** or **He** (default: "Xavier").
*  `--hidden_layers <int>`: Number of hidden layers in the network (default: 1).
*  `--neurons_nb <int>`: Number of neurons in each hidden layer (default: 2).
*  `--activations <list>`: Comma-separated list of functions for hidden and output layers - number of functions passed must be equal to hidden layers number + 1 (e.g., `relu,relu,sigmoid`).

### **2. Training & data management**
*  `--data_path <path>`: **(Required)** Path to the CSV dataset file.
*  `--target_column <name>`: **(Required)** Name of the column to predict.
*  `--test_fraction <float>`: Portion of data kept for validation (default: 0.2).
*  `--epochs <int>`: Number of training cycles (default: 140).
*  `--lr <float>`: Learning rate for weight updates (default: 0.2).
*  `--loss <MSE|BCE>`: Choice of loss function (default: "BCE").

### **3. File I/O**
*  `--save_model <path>`: Filename to save the model after execution .
*  `--load_model <path>`: File path to load weights from an existing model.

---

## **Building & running**

### **Compilation**
The project requires a C++ compiler with **OpenMP** support (such as GCC or Clang) for optimal training and feedforward.

```bash
clang++ -O3 -fopenmp main.cpp neural_network.cpp data_preprocessor.cpp file_data_operations.cpp display.cpp math_functions.cpp -o NeuralNetworkProjectFoCP
```
## Planned optimizations
### Data Structure Optimization
- **Currently:** The weights are stored in a nested `std::vector` (3D). While this was chosen for initial clarity and to match the mathematical representation of the network, it introduces a **pointer chasing** overhead.
- **Impact:** This structure limits the full potential of SIMD vectorization due to non-contiguous memory access.
- **To do:** flattening the weight tensors into a single, contiguous 1D array.
