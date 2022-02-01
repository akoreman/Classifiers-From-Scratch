# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt      

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix

class DistanceClassifier:
    '''
    Class to implement to implement the simple distance based classifier as defined in assingment 1.
    '''
    def train(self, inputArray, outputList, metricInput):
        '''
        Function to train the distance classifier on a dataset consisting of 256 length vectors.
        1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
        3rd parameter a string to choose the metric used to calculate the distances, choose from sklearn.metrics.pairwise.
        
        (list of vectors of length 256, list of ints, string) -> ()
        '''
        self.averageVecs = np.zeros((10,256))
        
        nrlist = np.zeros(10)
        
        for i in range(0, inputArray.shape[0]):
            self.averageVecs[int(outputList[i])] += inputArray[i]
            nrlist[int(outputList[i])] += 1
        
        for i in range(0,10):
            self.averageVecs[i] =  1/(nrlist[i]) * self.averageVecs[i]
         
        '''
        Calculate the distances and plot those.
        '''
        dist = np.zeros((10,10))
        
        for i in range(0,10):
            for j in range(0,10):
                dist[i][j] = pairwise_distances(self.averageVecs[j].reshape(1, -1), self.averageVecs[i].reshape(1, -1), metric = metricInput)[0][0]
                
                     
        fig, ax = plt.subplots()
        
        plt.imshow(dist, interpolation='nearest', cmap = 'Greys')
            
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, np.round(dist[i, j],1), ha="center", va="center", color="r")
        ax.set_ylabel('Class j')
        ax.set_xlabel('Class i') 
        
        plt.savefig("eucdist.pdf")
        
        plt.show()
          
        radius = np.zeros(10)
           
        for i in range(10):
            
            
            for j in range(10):
                if dist[j,i] > radius[i]:
                    radius[i] = dist[i,j]
                
    def test(self, inputArray, outputList, metricInput):  
        '''
        Function to use the classifier to classifry digits from a test set. Inputs the same as the train(...) function.
        Output a tuple with the first element the accuracy per digit and second element the confusion matrix as a double list.
        
        (list of vectors of length 256, list of labels, string) -> (list of length (10,2), list of length (10,10))
        '''
        results = np.zeros((10,2))
        
        predictions = []
        
        for inputs, label in zip(inputArray, outputList):
            results[int(label)][1] += 1
            
            #big number, used to compare to find minimum.
            minimum = 10000000
            minindex = 10
             
            for i in range(10):
                if pairwise_distances(inputs.reshape(1, -1), self.averageVecs[i].reshape(1, -1), metric = metricInput)[0][0] < minimum:
                    minindex = i

                    minimum = pairwise_distances(inputs.reshape(1, -1), self.averageVecs[i].reshape(1, -1), metric = metricInput)[0][0]
               
            predictions.append(minindex)
            
            if minindex == label:
                results[int(label)][0] += 1
                
        cm = confusion_matrix(predictions, outputList) 
                
        return [results,cm]      