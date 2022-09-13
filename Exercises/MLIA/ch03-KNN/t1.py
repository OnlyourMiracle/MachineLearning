dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
labels = ['no surfacing', 'flippers']
axis = 0
value = 1
for i in dataSet:
    if(i[axis] == value):
        print(i[axis])