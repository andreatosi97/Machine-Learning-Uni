import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE  # oversample
from imblearn.under_sampling import RandomUnderSampler  # undersample


def over_sampling(X, y, train_val_perc=0.8, train_perc=0.9, strat=1):
    # split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, train_size=train_val_perc, random_state=123
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=train_perc, random_state=123
    )

    # transform the dataset
    oversample = SMOTE(sampling_strategy=strat, random_state=123)
    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

    # define the model
    lr = LogisticRegression()

    # train
    lr.fit(X_train_over, y_train_over)

    # evaluate
    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)

    # F1 score is better for unbalanced situation
    train_score = f1_score(y_train, y_train_pred)
    val_score = f1_score(y_val, y_val_pred)

    return y_train_pred, y_val_pred, train_score, val_score


def under_sampling(X, y, train_val_perc=0.8, train_perc=0.9, strat=1):
    # split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, train_size=train_val_perc, random_state=123
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=train_perc, random_state=123
    )

    # transform the dataset
    undersampler = RandomUnderSampler(random_state=123, sampling_strategy=strat)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    # define the model
    lr = LogisticRegression()

    # train
    lr.fit(X_train_under, y_train_under)

    # evaluate
    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)

    # F1 score is better for unbalanced situation
    train_score = f1_score(y_train, y_train_pred)
    val_score = f1_score(y_val, y_val_pred)

    return y_train_pred, y_val_pred, train_score, val_score


## OUR EXAMPLE
# ad hoc/toy dataset: well acc with default lr but unbalanced data!
X_toy, y_toy = datasets.make_classification(
    n_samples=10000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.99],
    flip_y=0,
    random_state=1,
)

# split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_toy, y_toy, train_size=0.8, random_state=123
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, train_size=0.9, random_state=123
)

# both train and val set are unbalanced (see also with y_dummy = 0 vector)
print(f"Class 0: {np.sum(y_train == 0)}")
print(f"Class 1: {np.sum(y_train == 1)}")

print(f"Class 0: {np.sum(y_val == 0)}")
print(f"Class 1: {np.sum(y_val == 1)}")

# 	# scatter plot of examples by class label
# 	from collections import Counter
# 	from numpy import where

# 	counter = Counter(y_train)

# 	for label, _ in counter.items():
# 		row_ix = where(y_train == label)[0]
# 		plt.scatter(X_train[row_ix, 0], X_train[row_ix, 1], label=str(label))
# 	plt.legend()
# 	plt.show()

# 	also replot when transformed to check new situation

y_train_pred_over, y_val_pred_over, train_score_over, val_score_over = over_sampling(
    X_toy, y_toy, strat=0.05
)

(
    y_train_pred_under,
    y_val_pred_under,
    train_score_under,
    val_score_under,
) = under_sampling(X_toy, y_toy, strat=0.12)

print(val_score_over, val_score_under)
