#KNN.py


from numpy import *
import operator
from os import listdir


def classifyo(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}
        classLabelVector.append(int(listFormLine[-1]))  #pay attention
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classifyo(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with : %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount+=1.0
    print("the total error rates is : %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classifyo((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this persion:", resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
        return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('/root/Python/MachineLearning/Exercises/Ch02-KNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNmaeStr = trainingFileList[i]
        fileStr = fileNmaeStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('/root/Python/MachineLearning/Exercises/Ch02-KNN/trainingDigits/%s' %fileNmaeStr)
    testFileList = listdir('/root/Python/MachineLearning/Exercises/Ch02-KNN/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNmaeStr = testFileList[i]
        fileStr = fileNmaeStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/root/Python/MachineLearning/Exercises/Ch02-KNN/testDigits/%s' % fileNmaeStr)
        classifierResult = classifyo(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):errorCount += 1.0
    print("\nThe total number of errors is: %d" % errorCount)
    print("\nThe total error rate is: %f" % (errorCount/float(mTest)))