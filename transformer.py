"""
Transformer model implementation with only numpy
"""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Must be implemented in subclass")

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError("Must be implemented in subclass")

    @abstractmethod
    def update(self, learning_rate):
        pass


class Activations:

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return output


class MultiHeadAttention(Layer):

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize Q, K, V matrices
        self.W_Q = np.random.rand(d_model, d_model)
        self.W_K = np.random.rand(d_model, d_model)
        self.W_V = np.random.rand(d_model, d_model)

    def compute_heads(self, x):
        batch_size = x.shape[0]
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q = Q.reshape(batch_size, self.n_heads, -1, self.d_k)
        K = K.reshape(batch_size, self.n_heads, -1, self.d_k)
        V = V.reshape(batch_size, self.n_heads, -1, self.d_k)

        return Q, K, V

    def compute_attention(self, Q, K, V):
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        attention_weights = Activations.softmax(scores)
        output = attention_weights @ V
        return output, attention_weights

    def forward(self, x):
        Q, K, V = self.compute_heads(x)
        output, attention_weights = self.compute_attention(Q, K, V)
        output = output.reshape(
            x.shape[0], -1, self.d_model)  # Concatenate heads
        return output, attention_weights

    def backward(self, grad):
        grad = grad.reshape(
            grad.shape[0], self.n_heads, -1, self.d_k)
        grad_Q = grad @ self.W_Q.T
        grad_K = grad @ self.W_K.T
        grad_V = grad @ self.W_V.T
        grad_Q = grad_Q.reshape(grad.shape[0], -1)
        grad_K = grad_K.reshape(grad.shape[0], -1)
        grad_V = grad_V.reshape(grad.shape[0], -1)
        return grad_Q, grad_K, grad_V

    def update(self, learning_rate):
        # Update weights using gradient descent
        self.W_Q -= learning_rate * self.W_Q
        self.W_K -= learning_rate * self.W_K
        self.W_V -= learning_rate * self.W_V
