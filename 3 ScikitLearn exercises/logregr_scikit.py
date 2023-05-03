from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd


URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"


def lr_scikit(ds):
    # split into input X and output y elements
    data = ds.values
    X, y = data[:, :-1], data[:, -1]

    # partition in train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, train_size=0.75, random_state=123
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=0.9, random_state=123
    )

    # manual grid search
    C = [0.1, 0.5, 1, 5, 10, 15, 20, 50, 100]
    FI = [True, False]

    best_C = None
    best_fi = None
    best_train_acc = 0.0
    best_val_acc = 0.0

    for c in C:
        for fi in FI:
            # train
            clf_lr = LogisticRegression(C=c, fit_intercept=fi, max_iter=200)
            clf_lr.fit(X_train, y_train)

            # estimation (y_hat)
            y_train_pred_lr = clf_lr.predict(X_train)
            y_val_pred_lr = clf_lr.predict(X_val)

            tr_acc = accuracy_score(y_train, y_train_pred_lr)
            val_acc = accuracy_score(y_val, y_val_pred_lr)

            # look if it's best accuracy, so best hyperparameters
            if val_acc > best_val_acc:
                best_C = c
                best_fi = fi
                best_train_acc = tr_acc
                best_val_acc = val_acc

    return y_train_pred_lr, y_val_pred_lr, best_C, best_fi, best_train_acc, best_val_acc


### OUR EXAMPLE
# data
ds = pd.read_csv(URL, header=None)

(
    y_train_pred_lr,
    y_val_pred_lr,
    best_C,
    best_fi,
    best_train_acc,
    best_val_acc,
) = lr_scikit(ds)

print(f"Found the best model with C: {best_C}\t\t\tFit Intercept: {best_fi}")
print(f"Best training acc: {best_train_acc}\t\tBest val acc: {best_val_acc}")
