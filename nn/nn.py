from typing import List

import numpy as np

SIGMOID = lambda x: np.divide(1, 1 + np.exp(-x))

class NeuralNetwork():

    activation_fns = {
        'linear': (lambda x: x),
        'relu': (lambda x: np.maximum(0, x)),
        'sigmoid': SIGMOID,
        'tanh': (lambda x: np.tanh(x))
    }
    activation_derivative_fns = {
        'linear': (lambda x: 1),
        'relu': (lambda x: np.greater(x, 0).astype(int)),
        'sigmoid': (lambda x: SIGMOID(x) * (1 - SIGMOID(x))),
        'tanh': (lambda x: np.divide(1, np.cosh(x)))
    }

    loss_fns = {
        # Mean Squared Error
        'mse': (lambda y, y_hat: np.square(y - y_hat).mean()),

        # Binary Cross Entropy
        'bce': (lambda y, y_hat: np.sum(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat)) / y_hat.shape[0])
    }
    loss_derivative_fns = {
        # Mean Squared Error
        'mse': (lambda y, y_hat: (2 * (y_hat - y)).mean()),
        
        # Binary Cross Entropy
        'bce': (lambda y, y_hat: np.sum(np.divide(y_hat-y, y_hat*(1-y_hat))) / y_hat.shape[0])
    }

    def __init__(self, layers: np.int_, activations: List[str], loss: str):
        '''
        Initialize a Neural Network.
        @layers: array of integers representing the size of each layer (size: n * 1 vector)
        @activations: list of activation functions in `activation_fns` (size: (n-1) list)
        @loss: specifies the loss function to optimize the NN weights (one of: `mse` (mean-squared error), `bse` (binary cross-entropy))
        '''
        assert(layers.shape[0] == len(activations)+1)
        assert(loss in [*self.loss_fns])
        self.layers = layers
        self.activations = activations
        self.loss = loss

        self.W = [np.random.uniform(-1, 1, (self.layers[i], self.layers[i-1])) for i in range(1, len(self.layers))]
        self.b = [np.zeros((self.layers[i], 1)) for i in range(1, len(self.layers))]

    def _forward_propagation(self, inputs: np.ndarray):
        Z = np.empty((len(self.W),), dtype='object')
        A = np.empty((len(self.W) + 1,), dtype='object')
        A[0] = inputs
        for i in range(len(self.W)):
            Z[i] = np.matmul(self.W[i], A[i]) + self.b[i]
            A[i+1] = self.activation_fns[self.activations[i]](Z[i])
        return A, Z

    def _back_propagation(self, A: np.array, Z: np.array, y: np.array):
        dW = [None for i in range(1, len(self.layers))]
        db = [None for i in range(1, len(self.layers))]

        dA = self.loss_derivative_fns[self.loss](y, A[-1])
        for i in range(len(self.W)-1, -1, -1):
            dZ = dA * self.activation_derivative_fns[self.activations[i]](Z[i])
            m = A[i].shape[1]
            dW[i] = (1/m) * np.dot(dZ, A[i].T)
            db[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.W[i].T, dZ)
        return dW, db
    
    def _update_weights(self, dW, db, learning_rate):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - learning_rate * dW[i]
            self.b[i] = self.b[i] - learning_rate * db[i]

    def train(self, X, y, learning_rate = 0.005, num_iterations = 1000, output = False):
        for i in range(num_iterations):
            A, Z = self._forward_propagation(X)
            loss = self.loss_fns[self.loss](y, A[-1])
            if output:
                print(f'Iteration {i+1} Loss: {loss:.3f}')
            dW, db = self._back_propagation(A, Z, y)
            self._update_weights(dW, db, learning_rate)

if __name__ == '__main__':
    layers = np.array([2, 4, 4, 1])
    activations = ['relu', 'relu', 'sigmoid']

    nn = NeuralNetwork(layers, activations, 'mse')

    X_train = np.array([[1, 2], [2, 1]])
    y_train = np.array([1, 0])
    nn.train(X_train, y_train, output = True)
