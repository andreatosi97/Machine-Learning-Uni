import pandas as pd  # pandas manages our datasets (e.g., csv)
import numpy as np
import matplotlib.pyplot as plt  # get two graphic libraries
import seaborn as sns


DATA_URL = "https://raw.githubusercontent.com/cmdlinetips/data/master/cars.tsv"


# code the exercise as a function with the dataset as input
def np_closeform_lin_regr(data):
    # set up
    col_names = data.columns.values
    X = data[col_names[0]].values
    Y = data[col_names[1]].values
    X_mat = np.vstack([np.ones(len(X)), X]).T

    # solution computation
    beta_hat = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y)
    y_hat = X_mat.dot(beta_hat)
    error = np.sum((Y - y_hat) ** 2) / len(Y)

    return X, Y, y_hat, error


## Now we solve our case:
# import the dataset
cars = pd.read_csv(DATA_URL, sep="\t")

# plot the data
# bplot= sns.scatterplot(x = 'speed', y = 'dist', data=cars)    #define the type of plot
# bplot.axes.set_title("speed vs dist: Scatter Plot", fontsize=16)
# bplot.set_ylabel("Distances taken to stop (feet)", fontsize=16)
# bplot.set_xlabel("Speed (mph)", fontsize=16)

# solve
X, Y, y_hat, error = np_closeform_lin_regr(cars)

# see results
plt.scatter(X, Y)
plt.plot(X, y_hat, color="red")
plt.show()
print(error)
