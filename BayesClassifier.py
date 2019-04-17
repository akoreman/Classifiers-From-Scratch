# -*- coding: utf-8 -*-
import InputOutput as IO
import numpy as np
import matplotlib.pyplot as plt

class BayesClassifier:
    '''
    Class to implement the Bayes classifier as given in assignment 3.
    '''
    def __init__(self, digitA, digitB): 
        '''
        Initialize with 2 ints giving the 2 digits to classify.
        
        (int, int) -> ()
        '''
        self.digitA = digitA
        self.digitB = digitB
        
    def CalculateHeightWidth(self, reducedArray):
        '''
        Aux. function to calculate the height width ratio of a list of images coresponding to the same digit.
        
        (list of vectors of length 256) > (list of ints with same length as input list)
        '''
        reducedArray = [x.reshape((16,16)) for x in reducedArray]
        HeightWidth = np.zeros(len(reducedArray))
        
        #check where the digit begins from top and bottom to determine the height of the digit. Do the same for the transpose of the digit to determine the width.
        for i in range(len(reducedArray)):
            rowsums = [np.sum(x) for x in reducedArray[i]]
            columnsums = [np.sum(x) for x in reducedArray[i].transpose()]
          
            for j in range(16):
                if rowsums[j] > 1:
                    toprow = j
                    break
                
            for j in range(15,0,-1):
                if rowsums[j] > 1:
                    bottomrow = j
                    break
                            
            for j in range(16):
                if columnsums[j] > 1:
                    topcolumn = j
                    break
                
            for j in range(15,0,-1):
               if columnsums[j] > 1:
                    bottomcolumn = j
                    break
                
            height = toprow - bottomrow
            width = topcolumn - bottomcolumn
            
            HeightWidth[i] = height/width
        
        return HeightWidth
    
    
    def train(self, inputArray, outputList):
        '''
        Function to train the classifier.
        1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
        
        (list of vectors of length 256, list of ints) -> ()
        '''
        reducedInputA = IO.SliceDigit(inputArray, outputList, self.digitA)
        reducedInputB = IO.SliceDigit(inputArray, outputList, self.digitB)
                   
        featureA = self.CalculateHeightWidth(reducedInputA)
        featureB = self.CalculateHeightWidth(reducedInputB)  
                
        feature = np.concatenate((featureA, featureB))
            
        mi = np.amin(feature)
        ma = np.amax(feature)
                
        nbins = 12
        self.bins = np.linspace(mi,ma, nbins)
        
        '''
        Plot of the histograms
        '''
        fig, ax = plt.subplots()
   
        ax.set_ylabel('N')
        ax.set_xlabel('Height/Width')
        
        plt.hist(featureA, self.bins)
        plt.hist(featureB, self.bins)
        plt.savefig("hist.pdf")
        plt.show()
        
        #divide by zero is corrected for later so ignore those errors.          
        np.seterr(divide='ignore', invalid='ignore')
        
        conditionalsA = np.bincount(np.digitize(feature, self.bins),minlength = nbins + 1)/np.bincount(np.digitize(featureA, self.bins), minlength = nbins + 1)
        conditionalsB = np.bincount(np.digitize(feature, self.bins),minlength = nbins + 1)/np.bincount(np.digitize(featureB, self.bins), minlength = nbins + 1) 
                                              
        priorsA = featureA.size/(featureA.size + featureB.size)
        priorsB = featureB.size/(featureA.size + featureB.size)
                      
        PosteriorsA = conditionalsA * priorsA / (conditionalsA * priorsA + conditionalsB * priorsB)
        PosteriorsB = conditionalsB * priorsB / (conditionalsA * priorsA + conditionalsB * priorsB)
                  
        self.Classifier = np.zeros(len(PosteriorsA))
        
        for i in range(len(PosteriorsA)):
            if PosteriorsA[i] < PosteriorsB[i]:
                self.Classifier[i] = 0
            elif PosteriorsA[i] > PosteriorsB[i]:
                self.Classifier[i] = 1
                        
    def test(self, inputArray, outputList):
        '''
        Function to test the classifier on the test data.
        1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
        Outputs the accuracy for the digits chosen at initialisation.
        
        (list of vectors of length 256, list of ints) -> (double)
        '''
        reducedInputA = IO.SliceDigit(inputArray, outputList, self.digitA)
        reducedInputB = IO.SliceDigit(inputArray, outputList, self.digitB)
           
        featureA = self.CalculateHeightWidth(reducedInputA)
        featureB = self.CalculateHeightWidth(reducedInputB)

        ClassifiedA = np.digitize(featureA, self.bins)
        ClassifiedB = np.digitize(featureB, self.bins)

        correctA = 0
        correctB = 0
        
        for i in range(featureA.size):
            if self.Classifier[ClassifiedA[i]] == 0:
                correctA += 1

        for i in range(featureB.size):
            if self.Classifier[ClassifiedB[i]] == 1:
                correctB += 1
                
        accuracy = (correctA + correctB)/(featureA.size + featureB.size)
        
        return accuracy




















              