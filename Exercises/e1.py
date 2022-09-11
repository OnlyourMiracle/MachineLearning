import KNN
import importlib
importlib.reload(KNN)
'''
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')
normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
'''

KNN.handwritingClassTest()