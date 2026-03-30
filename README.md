# Breast Cancer Classification using Deep Learning from Scratch

This project implements a **Multi-Layer Perceptron (MLP)** with two hidden layers to classify breast cancer tumors as either **Malignant** or **Benign**. The entire neural network logic, including forward propagation and backpropagation, is built using only **NumPy**, without relying on high-level frameworks like TensorFlow or PyTorch.



## 🚀 Project Overview
The goal of this project is to demonstrate the inner workings of a neural network by implementing it from the ground up. We use the **Breast Cancer Wisconsin (Diagnostic) Dataset** to train the model.

### Key Features:
- **Architecture**: 2 Hidden Layers (16 and 8 neurons respectively).
- **Optimization**: Gradient Descent with manual Backpropagation.
- **Activation Function**: Sigmoid (with overflow protection).
- **OOP Design**: The model is encapsulated in a clean, reusable Python class.
- **Data Scaling**: Feature standardization using `StandardScaler`.

## 📊 Model Architecture
- **Input Layer**: 30 Features (radius, texture, perimeter, etc.)
- **Hidden Layer 1**: 16 Neurons + Sigmoid Activation
- **Hidden Layer 2**: 8 Neurons + Sigmoid Activation
- **Output Layer**: 1 Neuron (Binary Classification)



## 🛠️ Technologies Used
- **Python 3.x**
- **NumPy**: For matrix mathematics and weight updates.
- **Scikit-learn**: Used only for loading the dataset and preprocessing (scaling/splitting).
- **Matplotlib/Plotly**: For results visualization.

## 📈 Performance
After 5000 epochs of training, the model achieves high accuracy on the unseen test set:
- **Test Accuracy**: ~97-98%
- **Loss**: Smooth convergence over iterations.



## 💻 How to Run
1. Clone the repository.
2. Ensure you have `numpy` and `scikit-learn` installed.
3. Run the notebook or script to see the training progress and final accuracy.

```python
# Quick example of how the model is initialized
model = DeepNeuralNetwork(input_size=30, h1_size=16, h2_size=8, output_size=1)
model.train(X_train_scaled, y_train, epochs=5000)
