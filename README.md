# Modular C++ Neural Network

### **Overview**
This project is a high-performance, modular implementation of a feedforward neural network built in **C++**. Designed for both classification and regression tasks, it provides a pipeline for handling raw CSV data, automated preprocessing, and model state persistence.



---

## **Core Engineering Features**

*  **Automated Data Pipeline**: The `DataPreprocessor` class automatically detects column types.  It applies **Min-Max normalization** to numeric features and **One-Hot encoding** to categorical labels to prepare data for the network.
* **High-Performance Computing**: Optimized for speed using **OpenMP** to parallelize mathematical operations across multiple cores, reducing feedforward and training time for large datasets.
*  **Flexible Architecture**: Fully configurable network depth and layer width.  It supports multiple activation functions (**Sigmoid**, **ReLU**, **Tanh**) and loss functions (**MSE**, **BCE**) with native derivative implementations for backpropagation.
*  **Model Persistence**: Save or load trained models— weights, biases, and activations configurations to txt files .
* **Modularity**: Decoupled design between the math, data handling, and CLI parser ensures extensibility for future research and development.

---

## **Command Line Interface (CLI)**

 The application can be controlled via set of arguments parsed into a `Configuration` struct.

### **1. Model Configuration**
*  `--initialization <name>`: Weight initialization method—choose **Xavier** or **He** (default: "Xavier").
*  `--hidden_layers <int>`: Number of hidden layers in the network (default: 1).
*  `--neurons_nb <int>`: Number of neurons in each hidden layer (default: 2).
*  `--activations <list>`: Comma-separated list of functions for hidden and output layers - number of functions passed must be equal to hidden layers number + 1 (e.g., `relu,relu,sigmoid`).

### **2. Training & Data Management**
*  `--data_path <path>`: **(Required)** Path to the CSV dataset file.
*  `--target_column <name>`: **(Required)** Name of the column to predict.
*  `--test_fraction <float>`: Portion of data kept for validation (default: 0.2).
*  `--epochs <int>`: Number of training cycles (default: 140).
*  `--lr <float>`: Learning rate for weight updates (default: 0.2).
*  `--loss <MSE|BCE>`: Choice of loss function (default: "BCE").

### **3. File I/O & Persistence**
*  `--save_model <path>`: Filename to save the model after execution .
*  `--load_model <path>`: File path to load weights from an existing model.

---

## **Build & Run**

### **Compilation**
The project requires a C++ compiler with **OpenMP** support (such as GCC or Clang) for optimal training and feedforward.

```bash
clang++ -O3 -fopenmp main.cpp neural_network.cpp data_preprocessor.cpp file_data_operations.cpp display.cpp math_functions.cpp -o NeuralNetworkProjectFoCP
