# -*- coding: utf-8 -*-
'Python packages'
import numpy as np
import matplotlib.pyplot as plt

'Module created to input data and generate plots'
import InputOutput as IO

'Modules to run the classifiers'
import DistanceClassifier as DC
import BayesClassifier as BC
import Perceptron as P
import GradientDescent as GD

trainInput = IO.Rescale(IO.ReadCSV('train_in.csv'))
trainOutput = IO.ReadCSV('train_out.csv')

testInput = IO.Rescale(IO.ReadCSV('test_in.csv'))
testOutput = IO.ReadCSV('test_out.csv') 

xorInput = np.asarray([[0,0],[0,1],[1,0],[1,1]])
xorOutput = np.asanyarray([0,1,1,0])

'''
Here the classifiers as contructed in the other files are used to run the experiments and generate the plots for the report.
'''

'''
Assignment 1 & 2:
'''
dc = DC.DistanceClassifier()
dc.train(trainInput, trainOutput, 'euclidean')

resultsDcTrain = dc.test(trainInput, trainOutput, 'euclidean')
resultsDcTest = dc.test(testInput, testOutput, 'euclidean')

print("Distance classifier accuracy for train set : " + str(np.sum([x[0] for x in resultsDcTrain[0]])/np.sum([x[1] for x in resultsDcTrain[0]])))
print("Distance classifier accuracy for test set : " + str(np.sum([x[0] for x in resultsDcTest[0]])/np.sum([x[1] for x in resultsDcTest[0]])))

IO.PlotCM(resultsDcTrain[1], save = True, fileName = "distanceConfusionTrain")
IO.PlotCM(resultsDcTest[1], save = True, fileName = "distanceConfusionTest")

'''
Assignment 3:
'''
bc = BC.BayesClassifier(7,5)
bc.train(trainInput, trainOutput)

resultsBcTrain = bc.test(trainInput, trainOutput)
resultsBcTest = bc.test(testInput, testOutput)

print("Bayes classifier accuracy for train set for 5 & 7 : " + str(resultsBcTrain))
print("Bayes classifier accuracy for test set for 5 & 7: " + str(resultsBcTest))

bc = BC.BayesClassifier(1,4)
bc.train(trainInput, trainOutput)

resultsBcTrain = bc.test(trainInput, trainOutput)
resultsBcTest = bc.test(testInput, testOutput)

print("Bayes classifier accuracy for train set for 1 & 4: " + str(resultsBcTrain))
print("Bayes classifier accuracy for test set for 1 & 4: " + str(resultsBcTest))

'''
Assignment 4:
'''
p = P.Perceptron()
p.train(trainInput, trainOutput, 50)

resultsPTrain = p.test(trainInput, trainOutput)
resultsPTest = p.test(testInput, testOutput)

print("Perceptron accuracy for train set : " + str(np.sum([x[0] for x in resultsPTrain[0]])/np.sum([x[1] for x in resultsPTrain[0]])))
print("Perceptron accuracy for test set : " + str(np.sum([x[0] for x in resultsPTest[0]])/np.sum([x[1] for x in resultsPTest[0]])))

IO.PlotCM(resultsPTrain[1], save = True, fileName = "perceptronConfusionTrain")
IO.PlotCM(resultsPTest[1], save = True, fileName = "perceptronConfusionTest")

'''
Assignment 5:
'''

'''
The different activation functions are in commented out lines in the module GradientDescent.py
'''
gd = GD.GradientDescent()

weights = np.random.rand(9)
weights = weights * 2 - 1   

learningRate = 0.5

mseplot = []
accuracyplot = []

for i in range(1000):
    weights = weights - learningRate * gd.grdmse(weights)
    mseplot.append(gd.mse(weights))
    
    a = np.abs(np.around(gd.xor_net(0,0,weights)) - 0)
    b = np.abs(np.around(gd.xor_net(0,1,weights)) - 1)
    c = np.abs(np.around(gd.xor_net(1,0,weights)) - 1)
    d = np.abs(np.around(gd.xor_net(1,1,weights)) - 0)

    accuracyplot.append(1 - (a+b+c+d)/4)

'''
The plots for the MSE and the Accuracy
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('MSE')
ax1.set_xlabel('Iterations')
    
plt.plot(mseplot)

plt.savefig("xor_mse.pdf")
plt.show  

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Iterations')
    
plt.plot(accuracyplot)

plt.savefig("xor_acc.pdf")
plt.show      