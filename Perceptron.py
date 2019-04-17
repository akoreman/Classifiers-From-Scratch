# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.metrics import confusion_matrix

class Perceptron:
    '''
    Class to implement the perceptron algorithm as given in assignment 4.
    '''
    def train(self, inputArray, outputList, nIterations):
        '''
        Function to train the perceptron on a dataset consisting of 256 length vectors.
        1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
        3rd parameter a int to define how ofther the training needs to loop over the dataset.
        
        (list of vectors of length 256, list of ints, int, int) -> ()
        '''
        InputWeights = np.asarray([np.append(x,1) for x in inputArray])
        
        self.Weights = np.random.rand(10,257)
        self.Weights = self.Weights * 2 - 1
        
        '''
        Lists used for the plots of MSE and accuracy.
        '''
        trainPlot = []
        
        for _ in range(nIterations):
            score = 0
            error = 0
                               
            for inputs, label in zip(InputWeights,outputList):
                prediction = np.argmax(np.dot(self.Weights , inputs))
          
                if prediction == label:
                    score += 1
            
                labelVec = np.zeros((10,1))
                labelVec[int(label)] = 1
                
                predVec = np.zeros((10,1))
                predVec[int(prediction)] = 1
                            
                error += la.norm(labelVec - np.dot(self.Weights , inputs))**2

                self.Weights += (labelVec-predVec) * inputs
            
            trainPlot.append(score/inputArray.shape[0])
                   
        '''
        Plots for the MSE and Accuracy
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Iterations')
               
        plt.plot(trainPlot)
        plt.savefig("perceptron_accuracy.pdf")
                      
    def test(self, inputArray, outputList):
        '''
        Function to use the perceptron to classifry digits from a test set. 
        1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
        Output a tuple with the first element the accuracy per digit and second element the confusion matrix as a double list.
        
        (list of vectors of length 256, list of labels, string) -> (list of length (10, 2), list of length (10,10))
        '''
        InputWeights = np.asarray([np.append(a,1) for a in inputArray])
        results = np.zeros((10,2))
        
        predictions = []
        
        for inputs, label in zip(InputWeights,outputList):
            results[int(label)][1] += 1
            
            prediction = np.argmax(np.dot(self.Weights , inputs))
            predictions.append(prediction)
            
            if prediction == label:
                results[int(label)][0] += 1
        
        cm = confusion_matrix(predictions, outputList) 
                
        return [results,cm]