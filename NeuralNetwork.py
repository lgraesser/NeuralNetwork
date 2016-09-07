'''
MIT License

Copyright (c) [2016] [Laura Graesser]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This program allows the user to implement a general neural network, 
with an arbitrary number of layers and nodes per layer
It is intended to be an educational tool. No error checking is included so as
to simplify the code.

It was inspired by reading Michael Nielsen's excellent online book 
Neural Networks and Deep Learning, available at
http://neuralnetworksanddeeplearning.com/ so shares a similar structure to
his program

The main differences in functionality are that this is a vectorized implementation, 
this offers four different types of activation functions: Sigmoid, Relu, Leaky Relu, and Softmax,
it is possible to specify a different activation function for the hidden layers and output layer,
it offers different weight initializers, 
and operator overloading was used to simplify the feedforward step
'''

import numpy as np
np.set_printoptions(precision=8, suppress=True)

'''The Matrix class simplifies the appearance of matrix multiplication and addition,
   and was an excercise in operator overloading.
   If w, x and b are matrices of appropriate dimensions and W, X and B are
   the corresponding instances of the Matrix class then
   np.dot(w, x) + b is simplified to W * X + B

   Whilst it simplifies the appearance of the feedforward step, in practice this 
   did not turn out to be as useful as I had hoped because of the need to transpose
   matrices and carry out element wise operations.

   A better implementation would overload other operators to simplify the appearance
   of the transpose and element wise multiplication operations'''
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def __mul__(self, other):
        return Matrix(np.dot(self.matrix, other.matrix))
    
    def __add__(self, other):
        return Matrix(self.matrix + other.matrix)
    
    def __radd__(self, other):
        return Matrix(self.matrix + other.matrix)

'''Cost function comments:
   output is a 2D matrix containing the predicted output of a neural network 
   for a batch of training data
   
   y is 2D matrix containing the correct output for a batch of training data
   
   Each row corresponds to a data example, each column to an output node/feaure
   
   Element wise multiplication instead of a dot product is used to ensure that relevant
   values are squared when there are multiple examples and multiple output features'''           
    
class QuadraticCost:
    ''' Cost = 1 / 2n sum_x || y - output ||^ 2
        Can be used with any activation function in the output layer, however sigmoid is preferred''' 
    cost_type = "QuadraticCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        '''Cost function cost'''
        num_data_points = output.shape[0]
        diff = y - output
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w.matrix,w.matrix))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  np.sum(np.multiply(diff, diff)) / (2 * num_data_points) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''Derivative of the quadratic cost function'''
        return output - y
    
class CrossEntropyCost:
    ''' Cost = -1 / n sum_x (y * ln(output) + (1 - y)*ln(1- output))
        Should be used with a sigmoid output layer'''
    cost_type = "CrossEntropyCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        '''Cost function cost'''
        num_data_points = output.shape[0]
        interim = y * np.log(output) + (1 - y) * np.log(1 - output)
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w.matrix,w.matrix))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  (-1 / num_data_points) * np.sum(interim) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''Derivative of the cross entropy cost function
           ASSUMES that only sigmoid activation units are used in the output layer
           Derivative is not correct for other output layer activation functions, such as the ReLU
           Any activation function in the hidden layer can be used'''
        return output - y
    
class LogLikelihoodCost:
    ''' Cost = -1 / n ln output_c
        output_c is the output of the model for the correct answer, 
        this can be implemented by y * ln output_c since y will be 0 for the all but the correct answer
        Should be used with a softmax output layer'''
    cost_type = "LogLikelihoodCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        '''Cost function cost'''
        num_data_points = output.shape[0]
        interim = y * np.log(output)
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w.matrix,w.matrix))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  (-1 / num_data_points) * np.sum(interim) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''Derivative of the log likelihood cost function
           ASSUMES that only softmax activation units are used in the output layer
           Derivative is not correct for other output layer activation functions, such as the ReLU
           Any activation function in the hidden layer can be used'''
        return output - y    
    
class SigmoidActivation:
    @staticmethod
    def fn(x):
        '''Assumes x is an instance of the Matrix class'''
        return 1 / (1 + np.exp(-x.matrix))
    
    @staticmethod
    def prime(x):
        '''Assumes x is an instance of the Matrix class
           Derivative of the sigmoid function'''
        return np.multiply(SigmoidActivation.fn(x), (1 - SigmoidActivation.fn(x))) 
    
class ReluActivation:
    '''Should not be used in the output layer, only hidden layers'''
    @staticmethod
    def fn(x):
        '''Assumes x is an instance of the Matrix class'''
        y = np.copy(x.matrix)
        y = Matrix(y)
        ltzero_indices = y.matrix<0
        y.matrix[ltzero_indices] = 0
        return y.matrix
    
    @staticmethod
    def prime(x):
        '''Assumes x is an instance of the Matrix class'''
        ''' Derivative of the RELU function'''
        y = np.copy(x.matrix)
        y = Matrix(y)
        ltzero_indices = y.matrix<0
        other_indices = y.matrix>=0
        y.matrix[ltzero_indices] = 0
        y.matrix[other_indices] = 1
        return y.matrix

class LeakyReluActivation:
    '''Should not be used in the output layer, only hidden layers'''
    @staticmethod
    def fn(x):
        '''Assumes x is an instance of the Matrix class'''
        y = np.copy(x.matrix)
        y = Matrix(y)
        ltzero_indices = y.matrix<0
        y.matrix[ltzero_indices] = y.matrix[ltzero_indices] * 0.1
        return y.matrix
    
    @staticmethod
    def prime(x):
        '''Assumes x is an instance of the Matrix class'''
        ''' Derivative of the LRELU function'''
        y = np.copy(x.matrix)
        y = Matrix(y)
        ltzero_indices = y.matrix<0
        other_indices = y.matrix>=0
        y.matrix[ltzero_indices] = 0.1
        y.matrix[other_indices] = 1
        return y.matrix    
    
class SoftmaxActivation:
    @staticmethod
    def fn(x):
        '''Assumes x is an instance of the Matrix class'''
        '''Subtracting large constant from each of x values to prevent overflow'''
        y = np.copy(x.matrix)
        y = Matrix(y)
        max_per_row = np.amax(y.matrix, axis=1)
        max_per_row = max_per_row.reshape((max_per_row.shape[0], 1))
        y.matrix = y.matrix - max_per_row
        '''Adding small constant to prevent underflow'''
        exp_y = np.exp(y.matrix) + 0.001
        exp_y_sum = np.sum(exp_y, axis=1)
        exp_y_sum = np.reshape(exp_y_sum,(exp_y_sum.shape[0],1))
        return exp_y / exp_y_sum
    
    @staticmethod
    def prime(x):
        '''Assumes x is an instance of the Matrix class
           Derivative of the softmax function'''
        sftmax = SoftmaxActivation.fn(x) 
        return np.multiply(sftmax, (1 - sftmax))

class NeuralNet:
    def __init__(self, size, costfn=QuadraticCost, activationfnHidden=SigmoidActivation, \
                activationfnLast=SigmoidActivation):
        '''size = list of integers specifying the number of nodes per layer. Includes input and output layers. 
           e.g.(100,50,10) is a three layers network with one input layer, one hidden layer and one output layer
           costfn = cost function for the network. Should be an instance of one of the cost function classes
           activationfnHidden = activation function for all of the hidden nodes. Should be an instance
           of one of the activation function classes
           activationfnLast = activation function for the nodes in the last (output) layer. Should be an 
           instance of one of the activation function classes'''
        self.weights = []
        for a, b in zip(size[:-1], size[1:]):
            self.weights.append(np.zeros((a,b)))
        self.biases = []
        for b in size[1:]:
            self.biases.append(np.zeros((1, b)))
        self.layers = len(size)
        self.costfn = costfn
        self.activationfnHidden = activationfnHidden
        self.activationfnLast = activationfnLast
        
    def initialize_variables(self):
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = Matrix((np.random.uniform(-1, 1, size=w.shape) / np.sqrt(w.shape[0])))
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = Matrix(np.random.uniform(-1, 1, size=b.shape))
            i += 1
            
    def initialize_variables_normalized(self):
        '''Normalized initialization proposed by Glorot and Bengio, 2010
           Suggested for deep networks. Does not appear to be better than
           initialize_variables() for networks with 3 layers'''
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = Matrix(((np.random.uniform(-1, 1, size=w.shape) * np.sqrt(6))\
                                      / np.sqrt(w.shape[0] + w.shape[1])))
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = Matrix(np.random.uniform(-1, 1, size=b.shape))
            i += 1
            
    def initialize_variables_alt(self):
        '''Appears to be effective for shallow networks (~3 layers) with cross-entropy cost + ReLU hidden layers'''
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = Matrix((np.random.normal(size=w.shape) / w.shape[1]))
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = Matrix(np.random.normal(size=b.shape))
            i += 1

    def feedforward(self, data):
        '''Data = batch of input data, 2D matrix, examples x features
           Assumes data is structured as an m x n numpy array, examples x features
           Returns neural network output for this batch of data'''
        z = Matrix(data)
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            z = Matrix(self.activationfnHidden.fn(z * w + b))
        z = Matrix(self.activationfnLast.fn(z * self.weights[-1] + self.biases[-1]))
        return z.matrix
    
    def backprop(self, x, y, lmda):
        '''x = batch of input data, 2D matrix, examples x features
           y = corresponding correct output values for the batch, 2D matrix, examples x outputs
           lmda = regularization parameter
           z = weighted input to neurons in layer l. 
           a = activation of neurons in layer l.
           Layer 1 = input layer. No z value for layer 1, a_1 = x, i.e. no weights or activations
           
           Function returns the current cost for the batch and two lists of matrices:
           nabla_w = list of partial derivatives of cost w.r.t. weights per layer
           nabla_b = list of partial derivatives of cost w.r.t. biases per layer'''
        num_data_points = x.shape[0]
        z_vals = []
        a_vals = [x]
        
        ''' Feedforward: storing all z and a values per layer '''
        activation = Matrix(x)
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            z = activation * w + b
            z_vals.append(z.matrix)
            activation = Matrix(self.activationfnHidden.fn(z))
            a_vals.append(activation.matrix)
        z = activation * self.weights[-1] + self.biases[-1]
        z_vals.append(z.matrix)
        activation = Matrix(self.activationfnLast.fn(z))
        a_vals.append(activation.matrix)
              
        cost = self.costfn.compute_cost(a_vals[-1], y, lmda, self.weights)
        cost_prime = self.costfn.cost_prime(a_vals[-1], y)
        
        ''' Backprop: Errors per neuron calculated first starting at the last layer
            and working backwards through the networks, then partial derivatives 
            are calculated for each set of weights and biases
            deltas = neuron error per layer'''
        deltas = []
        
        if (self.costfn.cost_type=="QuadraticCost"):
            output_layer_delta = cost_prime * self.activationfnLast.prime(Matrix(z_vals[-1]))
        elif (self.costfn.cost_type=="CrossEntropyCost" or self.costfn.cost_type=="LogLikelihoodCost"):
            output_layer_delta = cost_prime
        else:
            print("No such cost function")
            exit(1)
        
        deltas.insert(0, output_layer_delta)
        
        for i in range(1,self.layers-1):
            interim = np.dot(deltas[0], (np.transpose(self.weights[-i].matrix)))
            act_prime = self.activationfnHidden.prime(Matrix(z_vals[-i-1]))
            delta = np.multiply(interim, act_prime)
            deltas.insert(0, delta)
        
        nabla_b = []
        for i in range(len(deltas)):
            interim = np.sum(deltas[i], axis=0) / num_data_points
            nabla_b.append(np.reshape(interim, (1, interim.shape[0])))
        
        nabla_w = []
        for i in range(0,self.layers-1):
            interim = np.dot(np.transpose(a_vals[i]), deltas[i])
            interim = interim / num_data_points
            nabla_w.append(interim)
        
        return cost, nabla_b, nabla_w
    
    def update_weights(self, nabla_w, nabla_b, learning_rate, lmda, num_data_points):
        '''nabla_w = list of partial derivatives of cost w.r.t. weights per layer
           nabla_b = list of partial derivatives of cost w.r.t. biases per layer
           learning_rate = learning rate hyperparamter, constrains size of parameter updates
           lmda = regularization paramter
           num_data_points = size of batch'''
        
        i = 0
        weight_mult = 1 - ((learning_rate * lmda) / num_data_points)
        for w, nw in zip(self.weights, nabla_w):
            self.weights[i].matrix = weight_mult * w.matrix - learning_rate * nw
            i += 1
        i = 0
        for b, nb in zip(self.biases, nabla_b):
            self.biases[i].matrix = b.matrix - learning_rate * nb
            i += 1
            
    def predict(self, x):
        '''x = batch of input data, 2D matrix, examples x features
           Function returns a 2D matrix of output values in one-hot encoded form'''
        
        output = self.feedforward(x)
        if (output.shape[1]==1):
            '''If only one output, convert to 1 if value > 0.5'''
            low_indices = output <= 0.5
            high_indices = output > 0.5
            output[low_indices] = 0
            output[high_indices] = 1
        else:
            '''Otherwise set maximum valued output element to 1, the rest to 0'''    
            max_elem = output.max(axis=1)
            max_elem = np.reshape(max_elem, (max_elem.shape[0], 1))
            output = np.floor(output/ max_elem)
        return output
    
    def accuracy(self, x, y):
        '''x = batch of input data, 2D matrix, examples x features
           y = corresponding correct output values for the batch, 2D matrix, examples x outputs
           Function returns % of correct classified examples in the batch'''
        prediction = self.predict(x)
        num_data_points = x.shape[0]
        if (prediction.shape[1]==1):
            result = np.sum(prediction==y) / num_data_points
        else:
            result = np.sum(prediction.argmax(axis=1)==y.argmax(axis=1)) / num_data_points
        return result
    
    def SGD(self, x, y, valid_x, valid_y, learning_rate, epochs, reporting_rate, lmda=0, batch_size=10, verbose=False):
        '''Stochastic Gradient Descent
           x = training data, 2D matrix, examples x features
           y = corresponding correct output values for the training data, 2D matrix, examples x outputs
           valid_x = validation data, 2D matrix, examples x features
           valid_y = corresponding correct output values for the validation data, 2D matrix, examples x outputs
           learning_rate = learning rate hyperparamter, constrains size of parameter updates
           epochs = number of iterations through the entire training dataset
           reporting_rate = rate at which to report information about the model's performance. 
           If the reporting rate is 10, then information will be printed every 10 epochs
           lmda = regularization paramter
           batch_size = batch size per parameter update. If batch size = 25 and there are 1000 examples in the 
           training data then there will be 40 updates per epoch
           verbose: parameter controlling whether to print additional information. Useful for debugging
           
           Function returns two lists contraining the training and validation cost per parameter update
           '''
        training_cost = []
        valid_cost = []
        num_data_points = batch_size
        total_data_points = x.shape[0]
        output = self.feedforward(x)
        cost = self.costfn.compute_cost(output,y,lmda,self.weights)
        accuracy = self.accuracy(x, y)
        valid_accuracy = self.accuracy(valid_x, valid_y)
        print("Training cost at start of training is %.5f and accuracy is %3.2f%%" % (cost, accuracy * 100))
        print("Validation set accuracy is %3.2f%%" % (valid_accuracy * 100))
        if (verbose==True):
            print("First 10 output values are:")
            print(output[0:10,:])
        
        for i in range(epochs):
            data = np.hstack((x,y))
            input_dims = x.shape[1]
            output_dims = y.shape[1]
            np.random.shuffle(data)
            batches = []
            nabla_w =[]
            nabla_b = []
            
            for k in range(0, (total_data_points - batch_size), batch_size):
                batch = data[k:(k+batch_size),:]
                batches.append(batch)
            num_batches = len(batches)
            for j in range(num_batches):
                batch_x = batches[j][:,:input_dims]
                batch_y = batches[j][:,input_dims:]
                if (batch_y.ndim == 1):
                    batch_y = np.reshape(batch_y, (batch_y.shape[0],1))
                cost, nabla_b, nabla_w = self.backprop(batch_x, batch_y, lmda)
                self.update_weights(nabla_w, nabla_b, learning_rate, lmda, num_data_points)

                '''Monitoring progress (or lack of...)'''
                training_cost.append(cost)
                valid_output = self.feedforward(valid_x)
                valid_c = self.costfn.compute_cost(valid_output,valid_y,lmda,self.weights)
                valid_cost.append(valid_c)
            
            if (i % reporting_rate == 0):
                output = self.feedforward(x)
                cost = self.costfn.compute_cost(output,y,lmda,self.weights)
                accuracy = self.accuracy(x, y)
                valid_accuracy = self.accuracy(valid_x, valid_y)
                print("Training cost in epoch %d is %.5f and accuracy is %3.2f%%" % (i, cost, accuracy * 100))
                print("Validation set accuracy is %3.2f%%" % (valid_accuracy * 100))
                if (verbose==True):
                    print("First 10 output values are:")
                    print(output[0:10,:])
                    print("Weight updates")
                    for i in range(len(self.weights)):
                        print(nabla_w[i])
                    print("Bias updates")
                    for i in range(len(self.biases)):
                        print(nabla_b[i])
                    print("Weights")
                    for i in range(len(self.weights)):
                        print(self.weights[i].matrix)
                    print("Biases")
                    for i in range(len(self.biases)):
                        print(self.biases[i].matrix)
                
        '''Final results'''
        output = self.feedforward(x)
        cost = self.costfn.compute_cost(output,y,lmda,self.weights)
        prediction = self.predict(x)
        accuracy = self.accuracy(x, y)
        valid_accuracy = self.accuracy(valid_x, valid_y)
        print("Final test cost is %.5f" % cost)
        print("Accuracy on training data is %3.2f%%, and accuracy on validation data is %3.2f%%" %
              (accuracy * 100, valid_accuracy * 100))  
        
        return training_cost, valid_cost