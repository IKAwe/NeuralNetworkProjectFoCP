C++ Modular Neural Network (OpenMP Optimized)

This project is a high-performance, modular C++ implementation of a feedforward neural network designed for classification and regression tasks using CSV datasets. It provides a complete end-to-end pipeline, from automated data preprocessing to model training and state persistence.
Core Engineering Features

    Automated Data Pipeline: The DataPreprocessor class identifies column types and automatically applies Min-Max normalization to numeric data and One-Hot encoding to categorical labels.

    High-Performance Computing: Utilizes OpenMP for parallelizing mathematical operations across multiple CPU cores, optimizing the training speed for larger datasets.

    Custom Math Engine: Includes native implementations of activation functions (Sigmoid, ReLU, Tanh) and loss functions (MSE, BCE) along with their derivatives.

    Model Persistence: Complete model states—including weights, biases, and layer configurations—can be exported and imported using robust file I/O operations.

    Modular Design: Developed with clean separation between the network logic, data handling, and mathematical functions to ensure extensibility for GSoC-level experimentation.

Command Line Interface (CLI)

The application is controlled via a comprehensive set of flags. The Configuration struct handles all runtime parameters .
1. Model Architecture & Initialization

    --initialization <name>: The method used to initialize weights—either Xavier or He (default: Xavier).

    --hidden_layers <int>: The number of hidden layers in the network (default: 1).

    --neurons_nb <int>: The number of neurons in each hidden layer (default: 2).

    --activations <list>: Comma-separated list (no spaces) of functions for each hidden and output layer (e.g., relu,relu,sigmoid).

2. Training & Data Management

    --data_path <path>: (Required) The path to the input CSV dataset.

    --target_column <name>: (Required) The name of the column you want the model to predict.

    --test_fraction <0-1>: The portion of the data reserved for validation (default: 0.2).

    --epochs <int>: Number of training iterations (default: 140).

    --lr <float>: Learning rate for backpropagation (default: 0.2).

    --loss <name>: Choose between MSE (L=21​(pred−actual)2) or BCE (L=−[y⋅log(p)+(1−y)⋅log(1−p)]) (default: BCE).

3. File I/O & Utility

    --save_model <path>: Specifies where to save the weights, biases, and activations after training.

    --load_model <path>: Loads a pre-existing model state from files.

    --help or -h: Displays the usage manual and exits.

Build & Performance
Compilation

The project requires a C++ compiler with OpenMP support.
Bash

g++ -O3 -fopenmp main.cpp neural_network.cpp data_preprocessor.cpp file_data_operations.cpp -o NNProject

Example Usage

Training a 3-layer network on the Iris dataset:
Bash

./NNProject --data_path iris.csv --target_column species --hidden_layers 3 --neurons_nb 8 --activations relu,relu,sigmoid --epochs 500 --lr 0.1 --loss BCE --save_model iris_v1

Technical Overview

    NeuralNetwork: Orchestrates forward passes and training using backpropagation.

    DataPreprocessor: Manages the fit-transform logic for data normalization.

    file_data_operations: Provides a robust abstraction for reading and writing multidimensional vectors to disk.