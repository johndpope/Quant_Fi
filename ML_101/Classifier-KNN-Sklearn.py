import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import numpy as np
from collections import Counter
import warnings

df = pd.read_csv('mnist_train.csv')

accuracies = []
passes = 100

for i in range(passes):
    indices = np.random.choice(len(df), 4000, replace=False)

    X = np.array(df.iloc[indices,1:])
    y = np.array(df.iloc[indices,0])

#SciKitLean Version of KNN
#split the data between a train set and a test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1, weights='distance')
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print("Accuracy {} pass: {:.2f}%".format(i, accuracy*100))
    accuracies.append(accuracy)

print("\nAccuracy after {} passes: {:.2f}%".format(passes, sum(accuracies)/len(accuracies)*100))
