# -*- coding: utf-8 -*-
import csv as csv
import numpy as np
import matplotlib.pyplot as plt

'''
This module contains functions used for input and output.
'''

def PlotDigit(array, size, save = False, fileName = ""):
    '''
    Function to plot the size x size vector of a dataset as a digit in a matplotlib heatplot.
    
    (list of length size x size, size) -> void.
    '''
    plt.imshow(array.reshape((size,size)), cmap='hot', interpolation='nearest')
    if save:
        plt.savefig(fileName + ".pdf")
    plt.show()
    
def ReadCSV(fileLocation):
    '''
    Function to read the CSV data files, input is the name of the CSV file, output nparray of vectors.
    
    (string) -> nparray of dimension # objects in dataset * length of each vector in dataset.
    '''
    with open(fileLocation,'r') as csvFile:
        array = []
        reader = csv.reader(csvFile)

        for row in reader:
            array.append(row)
            
        return np.array(array).astype(np.double)
       
def Rescale(inputArray):
    '''
    The values in the data files are in the range [-1,1], for some algorithms we want them to be in the range [0,1].
    
    (nparray of vectors) -> np array of vectors.
    '''
    for i in range(0, inputArray.shape[0]):
        inputArray[i] = (inputArray[i] + 1)/2
        
    return inputArray  

def SliceDigit(inputArrray, labelList, digit):
    '''
    Function to take from a dataset only the vectors belonging to a certain digit.
    1st and 2nd parameters are lists of equal length with data of 256 length vectors and scalars as labels.
    3rd parameter the digit that needs to be sliced out.
    
    (list of vectors of length 256, list of ints, int) -> (list of vectors of length 256)
    '''
    output = []
    
    for inputs, label in zip(inputArrray, labelList):
        if label == digit:
            output.append(inputs)
            
    return output

def PlotCM(conmatrix, save = False, fileName = ""):
    '''
    Function to plot the confusion matrix. Input the confusiom matrix as given by sklearn.metrics.confusion_matrix.
    Optional extra parameters are a bool to set if the image needs to be saved and string to specify the file name.
    
    (10x10 lists, optional: bool, optional: string) -> ()
    '''
    fig, ax = plt.subplots()

    plt.imshow(conmatrix, interpolation='nearest', cmap = 'Greys')
    
    ax.set_ylabel('Predicted label')
    ax.set_xlabel('Correct label')
    
    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, conmatrix[i,j], ha="center", va="center", color="r")
    
    if save:
        plt.savefig(fileName + ".pdf")
        
    plt.show()