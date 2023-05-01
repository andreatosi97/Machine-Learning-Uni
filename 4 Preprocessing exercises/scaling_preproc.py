from sklearn.datasets import load_wine    # load our target dataset
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler


def standardscale_lr(X, y, train_val_perc = 0.8, train_perc = 0.9):
    # split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size = train_val_perc, random_state = 123) 
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size = train_perc, random_state = 123)

    ## Default lr would give us error, then we scale
    # define the scaler
    scaler = StandardScaler()

    # fit the scaler on the training
    scaler.fit(X_train)

    # trasform the input
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #define a logistic regression
    lr = LogisticRegression()

    #fit
    lr.fit(X_train_scaled, y_train)

    #prediction
    y_train_pred = lr.predict(X_train_scaled)
    y_val_pred = lr.predict(X_val_scaled)

    #accuracy 
    tr_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    return y_train_pred, y_val_pred, tr_acc, val_acc


## OUR EXAMPLE
# load the dataset
dataset = load_wine()

# extract X and y
X = dataset.data        # 178 samples with 13 features
y = dataset.target      # has three classes

y_train_pred, y_val_pred, tr_acc, val_acc = standardscale_lr(X, y)

print(tr_acc, val_acc)
