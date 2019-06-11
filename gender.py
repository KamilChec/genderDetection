import numpy as np
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def classifier(model, x, y, test_data, ture_data):
    model.fit(x, y)
    prediction = model.predict(test_data)
    scores = cross_val_score(model, x, y, cv=5)
    print("Accuracy score: {}".format(accuracy_score(ture_data, prediction)))
    print('Cross validation accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    print("ROC AUC score: {}".format(roc_auc_score(ture_data, prediction)))


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
xNp = np.asarray(x)
yNp = np.asarray(y)
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
testNp = np.asarray(test)
trueNp = np.asarray(true)
lda = LinearDiscriminantAnalysis()
lr = LogisticRegression()
svm = SVC()

print("LinearDiscriminantAnalysis" + '-'*20)
classifier(lda, xNp, yNp, testNp, trueNp)
print("LogisticRegression" + '-'*20)
classifier(lr, xNp, yNp, testNp, trueNp)
print("SVC" + '-'*20)
classifier(svm, xNp, yNp, testNp, trueNp)
