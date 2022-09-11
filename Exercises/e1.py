import KNN
import importlib
importlib.reload(KNN)
'''
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')
normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
'''

print(KNN.img2vector('/root/Python/MachineLearning/Exercises/Ch02-KNN/testDigits/0_13.txt')[0, 0:31])