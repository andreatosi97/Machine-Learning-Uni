import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


## PLOTTING
# we def a function to plot decision boundary
# and then we'll see how it changes with different ker
def plot_discriminat_function(X, y, trained_model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # If we want a different scaling
    # h = (x_max / x_min)/100
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # X_plot = np.c_[xx.ravel(), yy.ravel()]
    x_span = np.linspace(x_min, x_max, 100)
    y_span = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_span, y_span)
    X_plot = np.c_[xx.ravel(), yy.ravel()]

    predicted = trained_model.predict(X_plot)
    predicted = predicted.reshape(xx.shape)

    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, predicted, alpha=0.5)
    plt.set_cmap("gist_rainbow")
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "r^")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")

    plt.xlim(xx.min(), xx.max())
    plt.title(trained_model)
    plt.show()


# also want to see accuracy
def compute_accuracy(trained_model, X_train, y_train, X_test, y_test):
    print("Accuracy on Train:", accuracy_score(y_train, trained_model.predict(X_train)))
    print("Accuracy on Test:", accuracy_score(y_test, trained_model.predict(X_test)))


## DATASET
X, y = datasets.make_classification(
    n_samples=700,
    n_features=2,
    random_state=124,
    n_redundant=0,
    n_informative=2,
    scale=10,
    shift=10,
)
# split, here we avoid val
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# scale
scl = StandardScaler()
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
# visualize the data
# fig = plt.figure(figsize=(7,5))
# plt.plot(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], 'r^')
# plt.plot(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], 'bs')
# plt.title('Random Classification Data with 2 classes')


## DIFFERENT KER MODELS
# polynomial of degree 2
print("Poly:")
svclassifier = SVC(kernel="poly", degree=2, coef0=1)
svclassifier.fit(X_train, y_train)
plot_discriminat_function(X_train, y_train, svclassifier)
compute_accuracy(svclassifier, X_train, y_train, X_test, y_test)
print()

# linear
print("Linear:")
svclassifier = SVC(kernel="linear")
svclassifier.fit(X_train, y_train)
plot_discriminat_function(X_train, y_train, svclassifier)
compute_accuracy(svclassifier, X_train, y_train, X_test, y_test)
print()

# rbf with different sigma
print("RBF:")
for sigma in [0.001, 0.1, 10, 100, 500]:
    svclassifier = SVC(kernel="rbf", gamma=sigma)
    svclassifier.fit(X_train, y_train)
    plot_discriminat_function(X_train, y_train, svclassifier)
    compute_accuracy(svclassifier, X_train, y_train, X_test, y_test)
    print()
