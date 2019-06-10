import numpy as np
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

x = []
y = []

with open(r'C:\Users\Kamil\Desktop\labSpec2\temp.csv', 'r') as file:
    for line in itertools.islice(file, 10 ** 4):
        fields = line.split(',')
        if fields[225] == '"Male"':
            y.append(1)
        else:
            y.append(0)
        del fields[225]
        del fields[230]
        del fields[259]
        x.append(list(map(float, fields[2:-1])))

clf = LinearDiscriminantAnalysis()
clf.fit(x, y)

true = []
test = []

with open(r'C:\Users\Kamil\Desktop\labSpec2\temp.csv', 'r') as file:
    for line in itertools.islice(file, 16000, 26000):
        fields = line.split(',')
        if fields[225] == '"Male"':
            true.append(1)
        else:
            true.append(0)
        del fields[225]
        del fields[230]
        del fields[259]
        test.append(list(map(float, fields[2:-1])))
prediction = clf.predict(test)
print(accuracy_score(true, prediction))
print(accuracy_score(true, prediction, normalize=False))
