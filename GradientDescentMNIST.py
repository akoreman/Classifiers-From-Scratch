# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import InputOutput as IO

trainInput = IO.Rescale(IO.ReadCSV('train_in.csv'))
trainOutput = IO.ReadCSV('train_out.csv')

'''
This module contains the partially finished code to use the gradient descent for the MNIST set.
Code is not completed so is not run from the main module and has to be run as a separate module.

Code is rough and without comments but functions mirror those as defined for the XOR gradient descent.
'''

def sig( x):
  return 1 / (1 + np.exp(-x))

def tanh( x):
    return (2 / (1 + np.exp(-2*x))) - 1
 
def mnist_net(array, weights):    
    inputs = np.append(array, 1)
   
    weights1,weights2 = np.split(weights, [7710])
    
    weights1 = np.reshape(weights1, (30,257))
    weights2 = np.reshape(weights2, (10,31))

    #hiddenlayer = np.asarray([sig(np.dot(x, inputs)) for x in weights1])
    hiddenlayer = np.asarray([tanh(np.dot(x, inputs)) for x in weights1])
    hiddenlayer = np.append(hiddenlayer, 1)
       
    #print(sig(np.dot(weights1[21], inputs)))
    #print(weights2.shape)
    
    #output = sig(np.dot(weights2, hiddenlayer))   
    output = tanh(np.dot(weights2, hiddenlayer))
    
    #print(output)
    
    return output
    
def mse(inputs, labels, weights): 
    output = []
    
    "think about this for a while"
    
    for inputs, label in zip(inputs,labels):
        labelVec = np.zeros(10)
        labelVec[int(label)] = 1
        
        #print(mnist_net(inputs, weights))
        a = (mnist_net(inputs, weights) - label)
        b= np.dot(a,a)
        
        #print(a)
        #print(b)
        
        output = np.append(output, b)
        
    return np.average(output)

def grdmse(inputs, labels, weights):
    eps = 10**(-3)
    
    gradient = np.zeros(weights.size)

    for i in range(weights.size):
        weightstemp = np.copy(weights)
        weightstemp[i] += eps
        #if i%100 == 0:
        #    print(i)
        gradient[i] = (mse(inputs, labels, weightstemp) - mse(inputs, labels, weights)) / eps
        
    return gradient
        
weights = np.random.rand(8020)
weights = weights * 2 - 1

learningRate = 5

#print(mnist_net(trainInput[1], weights))

mseplot = []

for i in range(2):
    print(i)
    
    random = np.random.choice(1707, 2)
    print(random)
    randomInput = np.asarray([trainInput[x] for x in random])
    randomOutput = np.asarray([trainOutput[x] for x in random])
    
    weights = weights - learningRate * grdmse(randomInput, randomOutput, weights)
    mseplot.append(mse(trainInput, trainOutput, weights))
   
plt.plot(mseplot)
plt.savefig("gde_mnist.pdf")
plt.show   
 
a = np.append(trainInput, 1)

predictions = []
results = np.zeros((10,2))

for inputs, label in zip(trainInput,trainOutput):
    
   
    results[int(label)][1] += 1
  
    
    prediction = np.argmax(mnist_net(inputs, weights))
    predictions.append(prediction)

    if prediction == label:
        results[int(label)][0] += 1

#cm = confusion_matrix(predictions, trainOutput) 
print(results)
print(np.sum([x[0] for x in results])/np.sum([x[1] for x in results]))