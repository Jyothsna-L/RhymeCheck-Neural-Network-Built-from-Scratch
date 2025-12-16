# RhymeCheck - A Neural Network from Scratch

## Project Overview

This project implements a neural network from scratch in Python to determine whether two English words rhyme or not. The system is built without using any machine learning libraries such as TensorFlow, PyTorch, or scikit-learn. All computations including forward propagation, loss calculation, and backpropagation are implemented manually using core Python constructs (loops, lists, conditionals, and basic math).

The primary goal of the project is not to achieve state-of-the-art rhyme detection, but to demonstrate a deep understanding of neural network fundamentals, including:
- Feature representation
- Weight and bias learning
- Gradient-based optimization
- Deterministic inference after training

## Problem Statement

Given two input words, the system predicts whether they rhyme by outputting:
- label = 1 Rhyme
- label = 0 Not Rhyme

Along with a confidence score between 0 and 100%.

## Key Characteristics

- Built entirely from first principles
- No external ML or NLP libraries
- Uses a fully connected feedforward neural network
- Supports interactive user input
- Fully inspectable weights, biases, and intermediate values

## Working

This system follows the standard structure of a feedforward neural network, implemented manually. It consists of four main stages:
1. Input encoding
2. Network architecture
3. Training (learning process)
4. Prediction (inference)

### Input Encoding

Since neural networks operate on numerical data, each input word is converted into a fixed-length numeric vector. Only the last four characters of each word are encoded, as rhyming depends primarily on word endings. Vowels are mapped to predefined numeric values, while consonants are assigned a constant value. If a word is shorter than four characters, it is padded with zeros. The encoded vectors of both words are concatenated, forming an 8-dimensional input vector.

### Network Architecture

The model is a fully connected feedforward neural network consisting of an input layer, one hidden layer, and an output layer. The input layer contains eight values representing the encoded word pair. The hidden layer consists of four neurons, each computing a weighted sum of the inputs followed by a sigmoid activation function. The output layer contains a single neuron that aggregates the hidden layer outputs and applies a sigmoid function to produce a probability score.

### Training

The network is trained using supervised learning with labeled word pairs. During training, each example is passed through the network using a forward pass, and the prediction error is computed using Mean Squared Error loss. Backpropagation is used to calculate gradients for each weight and bias, which are then updated using gradient descent. This process is repeated for multiple epochs until the loss stabilizes.

### Prediction

After training is complete, the model’s weights and biases remain fixed. During prediction, the encoded input is passed through the network using a forward pass only. The final sigmoid output represents the confidence that the two words rhyme. Based on predefined thresholds, this value is converted into a rhyme or non-rhyme label.

## Architecture 

The system uses a simple feedforward neural network with one hidden layer. The architecture is intentionally minimal to keep all computations transparent and easy to analyze.

The input layer receives an 8-dimensional vector formed by concatenating the encoded representations of two words. Each word contributes four numerical values, resulting in a fixed-size input regardless of word length.

This input is fully connected to a hidden layer containing four neurons. Each hidden neuron computes a weighted sum of the input values, adds a bias term, and applies a sigmoid activation function. The hidden layer is responsible for learning internal representations of phonetic patterns that may indicate rhyming behavior.

The output layer consists of a single neuron that takes the outputs of the hidden layer as input. It computes another weighted sum, adds a bias, and applies a sigmoid activation function. The resulting value lies between 0 and 1 and represents the model’s confidence that the input word pair rhymes.

Conceptually, the data flow through the network can be summarized as:

<img width="761" height="109" alt="image" src="https://github.com/user-attachments/assets/ea9e8314-2366-412c-89de-3096e9e9b5ee" />


<img width="551" height="403" alt="image" src="https://github.com/user-attachments/assets/f1045a11-71bf-44d8-b114-3a1876c1c153" />


This architecture enables the model to learn non-linear relationships between character patterns while remaining fully interpretable and built entirely from first principles.

## How to Run this Project

### Requirements

1. Python 3.x

This project is implemented in pure Python and does not require any external libraries beyond the standard library.

### Steps to run

1. Download the **rhymecheck.py** from the repository
2. Open a terminal or command prompt and navigate to the directory containing the file.
3. Run the program using: **python rhymecheck.py**
4. The model will first train automatically. During training, loss values are printed periodically to show learning progress.
5. After training, the program enters interactive mode. The user is prompted to enter two words, and the system outputs:
   - A confidence score
   - A rhyme or non-rhyme prediction
6. The user can continue testing multiple word pairs until choosing to exit.

**Credits:** Diagram created using [draw.io](https://app.diagrams.net/)  
**Credits:** Neural network visualization created using NN-SVG
