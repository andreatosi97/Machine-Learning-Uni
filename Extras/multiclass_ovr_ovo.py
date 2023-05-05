from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# let's create a dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=5,
    n_classes=3,
    random_state=1,
)

model = Perceptron(fit_intercept=True)
# define ovo strategy
ovo = OneVsOneClassifier(model)
# fit model
ovo.fit(X, y)
# make predictions
y_pred = ovo.predict(X)
print(accuracy_score(y, y_pred))

# same with One vs Rest
ovr = OneVsRestClassifier(model)
ovr.fit(X, y)
y_pred_ovr = ovr.predict(X)
print(accuracy_score(y, y_pred_ovr))
