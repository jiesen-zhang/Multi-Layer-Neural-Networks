"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.v = {}
        self.s = {}
        for i in range(1, num_layers + 1):
            shW = self.params["W" + str(i)].shape
            shb = self.params["b" + str(i)].shape
            self.v["dW" + str(i)] = np.zeros(shW) 
            self.s["dW" + str(i)] = np.zeros(shW) 
            self.v["db" + str(i)] = np.zeros(shb)
            self.s["db" + str(i)] = np.zeros(shb)
            # print("self.v[W + str(i)]", self.v["dW"+ str(i)].shape)
            # print("self.v[b + str(i)]", self.v["db"+ str(i)].shape)




    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # TODO: implement me

        return np.dot(X, W) + b # (N, D) dot (D, H1) + (H1,) = (N,H1) + (H1,)


    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me

        return np.maximum(0, X)

    def cross_entropy_loss(self, y, reg):
        """
        """
        # TODO: implement me
        a_n = self.outputs['a' + str(self.num_layers)]
        n_samples = a_n.shape[0]

        correct_logprobs = -np.log(self.outputs['a' + str(self.num_layers)][range(n_samples), y])
        data_loss = np.sum(correct_logprobs) / n_samples

        reg_loss = 0
        for i in range(1, self.num_layers + 1):
          reg_loss += ((1/2)*reg*np.sum(self.params['W' + str(i)] * self.params['W' + str(i)]))


        loss = data_loss + reg_loss        
        return loss

    def cross_entropy_gradient(self, y):
        """
        """
        # TODO: implement me

        n_samples = self.outputs['a' + str(self.num_layers)].shape[0]

        dscores = self.outputs['a' + str(self.num_layers)]
        dscores[range(n_samples), y] -= 1
        dscores /= n_samples

        return dscores



    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output data
        """
        # TODO: implement me
        X[X<=0] = 0
        X[X>1] = 1
        return X


    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        exp_X = np.exp(X - np.max(X))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        '''
        Li - linear output from i
        Ai - activation output from i
        '''

        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.


        # ensure output is (N, C)

        # scores = np.zeros(X.shape[0], self.params['W' + str(self.num_layers)].shape[1])
        # assert(scores.shape[1] == self.output_size)

        self.outputs['a' + str(0)] = X

        for i in range(1, self.num_layers):
          self.outputs['z' + str(i)] = self.outputs['a' + str(i - 1)].dot(self.params['W' + str(i)]) + self.params['b' + str(i)]
          self.outputs['a' + str(i)] = self.relu(self.outputs['z' + str(i)])

        self.outputs['z' + str(self.num_layers)] = self.outputs['a' + str(self.num_layers - 1)].dot(self.params['W' + str(self.num_layers)]) + self.params['b' + str(self.num_layers)]
        # self.outputs['z' + str(self.num_layers)] = self.linear(self.outputs['a' + str(self.num_layers - 1)], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        self.outputs['a' + str(self.num_layers)] = self.softmax(self.outputs['z' + str(self.num_layers)])

        scores = self.outputs['a' + str(self.num_layers)]
        return scores


    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.

        Note: both gradients and loss should include regularization.

        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        #loss
        loss = self.cross_entropy_loss(y, reg)

        self.gradients['dhidden' + str(self.num_layers)] = self.cross_entropy_gradient(y)

        n_samples = self.outputs['a' + str(self.num_layers)].shape[0]

        for i in range(self.num_layers, 0, -1):
          self.gradients['W' + str(i)] = np.dot(self.outputs['a' + str(i-1)].T, self.gradients['dhidden' + str(i)])
          self.gradients['b' + str(i)] = np.sum(self.gradients['dhidden' + str(i)], axis=0)

          self.gradients['dhidden' + str(i - 1)] = np.dot(self.gradients['dhidden' + str(i)], self.params['W' + str(i)].T)
          self.gradients['dhidden' + str(i - 1)][self.outputs['a' + str(i - 1)] <= 0] = 0

          self.gradients['W' + str(i)] += reg * self.params['W' + str(i)]

        return loss





    def update(self,lr: float = 0.001,b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, opt: str = "SGD", t: int = 1):
        """Update the parameters of the model using the previously calculated
        gradients.

        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        if opt =="SGD":
          for i in range(1, self.num_layers):
            self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
            self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]

        elif opt == "Adam":

          # self.v = v
          # self.s = s
          v_corrected = {}
          s_corrected = {}
          for i in range(1, self.num_layers):
            self.v['dW' + str(i)] = (b1 * self.v['dW' + str(i)] + (1 - b1) * self.gradients['W' + str(i)]) 
            self.v['db' + str(i)] = (b1 * self.v['db' + str(i)] + (1 - b1) * self.gradients['b' + str(i)]) 


            v_corrected['dW' + str(i)] = self.v['dW' + str(i)] / (1 - np.power(b1,t))
            v_corrected['db' + str(i)] = self.v['db' + str(i)] / (1 - np.power(b1,t))


            self.s['dW' + str(i)] = (b2 * self.s['dW' + str(i)] + (1 - b2) * np.power(self.gradients['W' + str(i)], 2)) 
            self.s['db' + str(i)] = (b2 * self.s['db' + str(i)] + (1 - b2) * np.power(self.gradients['b' + str(i)],2)) 

            s_corrected['dW' + str(i)] = self.s['dW' + str(i)] / (1 - np.power(b2, t))
            s_corrected['db' + str(i)] = self.s['db' + str(i)] / (1 - np.power(b2, t))
            
            # self.params['W' + str(i)] -= (lr * self.v['dW' + str(i)]) / (np.sqrt(self.s['dW' + str(i)]) + eps)
            # self.params['b' + str(i)] -= (lr * self.v['db' + str(i)]) / (np.sqrt(self.s['db' + str(i)]) + eps)

            self.params['W' + str(i)] -= (lr * v_corrected['dW' + str(i)]) / np.sqrt(s_corrected['dW' + str(i)] + eps)
            self.params['b' + str(i)] -= (lr * v_corrected['db' + str(i)]) / np.sqrt(s_corrected['db' + str(i)] + eps)

        return