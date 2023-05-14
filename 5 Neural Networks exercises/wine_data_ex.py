# sklearn import
from sklearn.datasets import load_wine  # load our target dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp_funct import mlp_def_train


## DATASET
# load the dataset
dataset = load_wine()
# extract X and y
X = dataset.data
# print(X.shape)
y = dataset.target
# split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, train_size=0.8, random_state=123
)

# check balance
# print(Counter(y))
# check scaling
# print(X.min(axis = 0))
# print(X.max(axis = 0))
# print(X.mean(axis = 0 ))

## TEST
scaler = StandardScaler()
scaler.fit(X_train_val)  # remember, always training!
X_train_scl = scaler.transform(X_train_val)
X_test_scl = scaler.transform(X_test)

## GRID SEARCH
nlayers = [1, 2, 3, 4]
for nl in nlayers:
    m, h, test_res = mlp_def_train(
        X_train_scl, y_train_val, X_test_scl, y_test, n_layers=nl, activation_f="relu"
    )

    print(
        f"N layers:{nl}\t\tTrain ACC:{h.history['accuracy'][-1]:.8f}\t\tVal ACC:{h.history['val_accuracy'][-1]:.8f}"
    )
    print(f"Test results:\n\t - Loss: {test_res[0]} \n\t - Accuracy: {test_res[1]}%\n")
