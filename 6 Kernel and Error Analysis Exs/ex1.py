import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


## DATASET
# Define Dataset for some next exercises
digits = datasets.load_digits()
# Split data into 80% train+validation and 20% test subsets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, shuffle=False
)

## TRAIN
svc_grid_params = {"C": (0.1, 1.0, 10)}
svc_clf = SVC(kernel="rbf")
svc_v2 = GridSearchCV(svc_clf, svc_grid_params, n_jobs=-1, cv=5)
svc_v2.fit(X_train_val, y_train_val)
y_test_pred = svc_v2.predict(X_test)
# See that we've a good score
print(accuracy_score(y_test_pred, y_test))

## CONFUSION MATRIX
cm = confusion_matrix(y_test, y_test_pred, labels=svc_v2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc_v2.classes_)
disp.plot()
plt.show()

## CONTROL REASONABLE ERRORS FROM CONFUSION MATRIX

# find the index of the examples of class 4 predicted as class 9
# np.all returns the elements of the input vector that satisfy all conditions
indices = np.all([(y_test == 4), (y_test_pred == 9)], axis=0)
_, axes = plt.subplots(nrows=1, ncols=sum(indices == True), figsize=(7, 2))

for ax, image, label, pred_label in zip(
    axes, X_test[indices], y_test[indices], y_test_pred[indices]
):
    ax.set_axis_off()
    ax.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Test: %i\nPred: %i" % (label, pred_label))
plt.show()
