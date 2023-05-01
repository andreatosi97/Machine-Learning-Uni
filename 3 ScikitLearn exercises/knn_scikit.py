import numpy as np
import matplotlib.pyplot as plt
# scikit
from sklearn.datasets import make_moons                 # to create fake dataset
from sklearn.neighbors import KNeighborsClassifier      # implemented K-NN
from sklearn.model_selection import train_test_split    # split train,validation and test st
from sklearn.metrics import accuracy_score              # function for accuracy



def knn_scikit(X,y):
    # divide training and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.3, random_state=123)

    accuracy_values_train = []
    accuracy_values_test = []
    k_values = range(1, 80)
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        # train
        model.fit(X_train, y_train)
        # prediction
        y_pred_train = model.predict(X_train)   
        y_pred_test = model.predict(X_test)
        # compute accuracy
        accuracy_values_train.append(accuracy_score(y_pred_train, y_train))
        accuracy_values_test.append(accuracy_score(y_pred_test, y_test))
    
    return y_pred_train, y_pred_test, accuracy_values_train, accuracy_values_test



### OUR EXAMPLE
# plot data
X, y = make_moons(n_samples=750, noise=0.3, random_state=123)   # dataset with random seed: X = 2D pts, y = layers
fig = plt.figure(figsize=(7,5))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')    # pts with y=0 are red (X[:,0] means first component, X[:,1] second)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')    # pts with y=1 are blue
plt.show()

# execute the function
y_pred_train, y_pred_test, accuracy_values_train, accuracy_values_test = knn_scikit(X,y)

# plot accuracy value wrt K
k_values = range(1,80)
fig = plt.figure(figsize=(7,5))
plt.plot(k_values, accuracy_values_train, label="train")
plt.plot(k_values, accuracy_values_test, label="test")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()