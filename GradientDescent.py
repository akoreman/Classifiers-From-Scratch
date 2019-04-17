# -*- coding: utf-8 -*-
import numpy as np

class GradientDescent:
    def sig(self, x):
        '''
        Applies the sigmoid function on the input.
        
        (int) -> (int)
        '''
        return 1 / (1 + np.exp(-x))
  
    def relu(self, x):
        '''
        Applies the linear rectifier on the input.
        
        (int) -> (int)
        '''
        if x <= 0:
            return 0
        elif x > 0:
            return x
        
    def tanh(self, x):
        '''
        Applies the hyperbolic tangent function on the input.
        
        (int) -> (int)
        '''
        return (2 / (1 + np.exp(-2*x))) - 1
     
    def xor_net(self, x1,x2,weights):
        '''
        Function to give a prediction for the XOR gate using the weights.
        1st and 2nd parameters the inputs x and y.
        3rd parameter the weights given as a vector of length 9.
        
        (int, int, vector length 9) -> (int)
        '''
        inputs = np.asarray([x1,x2])
        inputs = np.append(inputs, 1)
        
        weights = np.reshape(weights, (3,3))
        
        'commented out lines change the activation function'
        #hiddenlayer = np.asarray([self.sig(np.dot(weights[0], inputs)),self.sig(np.dot(weights[1], inputs)),1])   
        #hiddenlayer = np.asarray([self.relu(np.dot(weights[0], inputs)),self.sig(np.dot(weights[1], inputs)),1]) 
        hiddenlayer = np.asarray([self.tanh(np.dot(weights[0], inputs)),self.sig(np.dot(weights[1], inputs)),1]) 
        
        #output = self.sig(np.dot(weights[2], hiddenlayer))   
        #output = self.relu(np.dot(weights[2], hiddenlayer))
        output = self.tanh(np.dot(weights[2], hiddenlayer))
     
        return output
        
    def mse(self, weights):
        '''
        Function to calculate the mean squared error for the XOR model for a set of weights.
        
        (vector of length 9) -> (double)
        '''
        x = np.asarray([[0,0],[0,1],[1,0],[1,1]])
        y = np.asanyarray([0,1,1,0])
        
        output = []
        
        for xs, label in zip(x,y):
            output = np.append(output, (self.xor_net(xs[0],xs[1], weights) - label)**2)
            
        return np.average(output)
    
    def grdmse(self, weights):
        '''
        Function to calculate the gradient for a set of weights to minimize the MSE.
        Outputs a vector with the same length as the input weights.
        
        (vector) -> (vector with same length as input)
        '''
        eps = 10**(-3)
        
        gradient = np.zeros(9)
    
        for i in range(weights.size):
            weightstemp = np.copy(weights)
            weightstemp[i] += eps
    
            gradient[i] = (self.mse(weightstemp) - self.mse(weights)) / eps
            
        return gradient
        