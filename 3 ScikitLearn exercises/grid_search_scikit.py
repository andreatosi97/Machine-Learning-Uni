from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd


def grid_search_scikit(ds, model, grid):
    # split into input X and output y elements
    data = ds.values
    X, y = data[:, :-1], data[:, -1]
    # partition in train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                  train_size = 0.75, random_state = 123)

    # grid-search object
    clf = GridSearchCV(estimator= model, param_grid=grid, 
                        cv = 10, scoring = "accuracy")

    # train/fit the model
    clf.fit(X_train_val, y_train_val) # we do not use the train-validation split strategy since it is included in the CV procedure

    # prediction and performance on test
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # return best parameters and performance
    return  clf.best_params_, clf.best_score_, y_test_pred, test_acc



### OUR EXAMPLE
# data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
ds = pd.read_csv(url, header = None)
# grid
param_grid_test = {
    'C': [0.1, 0.5, 1, 5, 10, 15, 20, 50, 100],
    'fit_intercept': [True, False]
}
# target classifier
lr = LogisticRegression(max_iter = 200)

best_par, best_score, y_test_pred, test_acc = grid_search_scikit(ds, lr, param_grid_test)

print(best_par, '\n', best_score, '\n',  y_test_pred, '\n', test_acc)