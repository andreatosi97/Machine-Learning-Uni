import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# keras for NN
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.framework.random_seed import set_random_seed
from keras.callbacks import EarlyStopping


## FUNCTION TO TRACK TIMES AND ACCURACIES
# give us the mean time to train, mean time to predict,
# mean train acc, mean test acc over 5 iterations
def test_algorithm(learning_alg, X_train, y_train, X_test, y_test):
    # empty list to keep track of the running iterations and accuracy values of each repetition
    train_time_iter, pred_time_iter, train_accuracy_iter, test_accuracy_iter = (
        [],
        [],
        [],
        [],
    )

    for _ in range(5):
        start_time = time.time()  # get the starting time
        learning_alg.fit(X_train, y_train)
        end_time = time.time()  # get the ending time
        train_time_iter.append(end_time - start_time)

        start_time = time.time()  # get the starting time
        y_train_pred = learning_alg.predict(X_train)
        y_test_pred = learning_alg.predict(X_test)
        end_time = time.time()  # get the ending time
        pred_time_iter.append(end_time - start_time)

        train_accuracy_iter.append(accuracy_score(y_train, y_train_pred))
        test_accuracy_iter.append(accuracy_score(y_test, y_test_pred))

    return (
        np.mean(train_time_iter),
        np.mean(pred_time_iter),
        np.mean(train_accuracy_iter),
        np.mean(test_accuracy_iter),
    )


## DATASET
digits = datasets.load_digits()
# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, shuffle=False
)
# scale the data
scl = StandardScaler().fit(X_train)
X_train_scl = scl.transform(X_train)
X_test_scl = scl.transform(X_test)

num_classes = len(set(y_train))
num_features = X_train.shape[1]


## COLLECTS VALUES FOR DIFFERENT ALG/MODEL
# initialize the lists collecting the stats
tr_time_list, pr_time_list, tr_acc_list, te_acc_list, model_name_list = (
    [],
    [],
    [],
    [],
    [],
)

# Perceptron
model_name_list.append("Perceptron")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    Perceptron(), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

# Logistic Regression
model_name_list.append("Logistic Regression")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    LogisticRegression(), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

# SVM with different ker
model_name_list.append("SVM Linear Kernel")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    SVC(kernel="linear"), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

model_name_list.append("SVM Polynomial Kernel d=2")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    SVC(kernel="poly", degree=2), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

model_name_list.append("SVM Polynomial Kernel d=3")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    SVC(kernel="poly", degree=3), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

# Tree
model_name_list.append("Decision Tree")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    tree.DecisionTreeClassifier(criterion="entropy"),
    X_train_scl,
    y_train,
    X_test_scl,
    y_test,
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

# k-NN
model_name_list.append("3-NN")
train_time, pred_time, train_accuracy, test_accuracy = test_algorithm(
    KNeighborsClassifier(n_neighbors=3), X_train_scl, y_train, X_test_scl, y_test
)
tr_time_list.append(train_time)
pr_time_list.append(pred_time)
tr_acc_list.append(train_accuracy)
te_acc_list.append(test_accuracy)

# Print results
for i in range(len(tr_time_list)):
    print(
        f"{model_name_list[i]}\n   Training time: {tr_time_list[i]}\n   Prediction time: {pr_time_list[i]}"
    )
    print(
        f"   Accuracy on train: {tr_acc_list[i]}\n   Accuracy on test {te_acc_list[i]}"
    )
print()

## CHECK TIME vs ACC PLOT
# Checks if you have the right amount of info
expected_list_length = 7
assert len(tr_acc_list) == expected_list_length
assert len(te_acc_list) == expected_list_length
assert len(tr_time_list) == expected_list_length
assert len(pr_time_list) == expected_list_length
assert len(model_name_list) == expected_list_length

# color map of acc vs train time
colors = cm.rainbow(np.linspace(0, 1, len(model_name_list)))
for x, y, c, m in zip(te_acc_list, tr_time_list, colors, model_name_list):
    plt.scatter(x, y, color=c, label=m)
plt.legend()
plt.xlabel("Test Accuracy")
plt.ylabel("Training Time [s]")
plt.show()

# color map of acc vs prediction time
colors = cm.rainbow(np.linspace(0, 1, len(model_name_list)))
for x, y, c, m in zip(te_acc_list, pr_time_list, colors, model_name_list):
    plt.scatter(x, y, color=c, label=m)
plt.legend()
plt.xlabel("Test Accuracy")
plt.ylabel("Prediction Time [s]")
plt.show()


## SAME FUNCTION FOR A THREE LAYER NN
# we set random seeds to check some results
np.random.seed(5)
set_random_seed(42)

# categorization
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


def test_nn(
    X_train,
    y_train,
    num_features,
    first_layer_units,
    second_layer_units,
    hid_activation,
):
    es = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=1)
    train_time, pred_time, train_accuracy, test_accuracy = ([], [], [], [])

    for _ in range(5):
        model = Sequential()
        model.add(
            Dense(
                input_dim=num_features,
                units=first_layer_units,
                activation=hid_activation,
            )
        )
        model.add(Dense(units=second_layer_units, activation=hid_activation))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )

        start_time = time.time()
        h = model.fit(
            X_train,
            y_train,
            epochs=300,
            verbose=0,
            validation_split=0.20,
            callbacks=[es],
        )
        end_time = time.time()
        train_time.append(end_time - start_time)

        start_time = time.time()
        train_accuracy.append(model.evaluate(X_train_scl, y_train_cat, verbose=0)[1])
        test_accuracy.append(model.evaluate(X_test_scl, y_test_cat, verbose=0)[1])
        end_time = time.time()
        pred_time.append(end_time - start_time)

    return train_time, pred_time, train_accuracy, test_accuracy


train_time, pred_time, train_accuracy, test_accuracy = test_nn(
    X_train_scl, y_train_cat, num_features, 2 * num_features, num_features / 2, "relu"
)

# print results
print("\nAverage Model Test Accuracy: %.4f%%" % (np.mean(test_accuracy)))
print("Times:\n\t", train_time)
print("\t", pred_time)
print("Acc:\n\t", train_accuracy)
print("\t", test_accuracy)
print(
    "Difference between fastest and shortest training times:",
    max(train_time) - min(train_time),
)
print(
    "Difference bwteen best and worst test results:",
    max(test_accuracy) - min(test_accuracy),
)
