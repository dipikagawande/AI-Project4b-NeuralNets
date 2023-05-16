from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt
import numpy as np

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)


##---------------------------------------------------------------##
## Testing Q4
testPenData()

##---------------------------------------------------------------##
## Testing Q5
penAccurList = []
carAccurList = []

for iter in range(5):
	penTest = testPenData()[1]
	penAccurList.append(penTest)
        
	carTest = testCarData()[1]
	carAccurList.append(carTest)
    
print("---Q5---")
print("Pen Test Accuracy", penAccurList,  
      "\nmax: ", np.max(penAccurList),
      "\nmean: ", np.mean(penAccurList),
      "\nsd: ", np.std(penAccurList))

print("Car Test Accuracy:", carAccurList,  
      "\nmax: ", np.max(carAccurList),
      "\nmean: ", np.mean(carAccurList),
      "\nsd: ", np.std(carAccurList))

## EOF