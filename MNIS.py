import scipy.io 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


mnist = scipy.io.loadmat('mnist-original.mat')
X, y = mnist["data"], mnist["label"]
X.shape
y.shape

#Prepair data
X=X.transpose()
y=y.transpose()
y=y.ravel()
#Split to training set and testing set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Create randomness?????
from sklearn.linear_model import SGDClassifier

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#Testing SGDClassifier:
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# cross_val_score(sgd_clf, X_train, y_train_5, cv=3)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
#Testing KNeighborsClassifier model:
knc_clf = KNeighborsClassifier(n_neighbors=3)
knc_clf.fit(X_train, y_train_5)
# creating odd list of K for KNN
myList = list(range(1,5))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knc_clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knc_clf, X_train, y_train_5, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
# pred = knc_clf.predict(X_test[32000])
# y_train_pred_KNC = cross_val_predict(knc_clf, X_train, y_train_5, cv=3
confusion_matrix(y_train_5, y_train_pred_KNC)

#Testing RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train_5)

y_train_pred_RFC = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred_RFC)


# from sklearn.base import BaseEstimator



# cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

