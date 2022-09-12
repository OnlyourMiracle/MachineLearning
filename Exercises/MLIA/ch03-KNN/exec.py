import trees
import importlib
importlib.reload(trees)
myDat, labels = trees.createDataSet()
print(myDat)