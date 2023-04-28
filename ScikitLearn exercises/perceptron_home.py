import matplotlib.pyplot as plt
from sklearn import datasets


def perceptron_np_train(X_mat, y, theta, max_epochs):
	m = X_mat.shape[0] 		# number of examples in the training set
	num_epochs = 0
	num_errors = 1
	while num_epochs <= max_epochs and num_errors > 0: # we keep iterating over the data until 
														# we make no mistake or we reach the maximum number of iterations
		num_epochs = num_epochs + 1
		num_errors = 0
		for i in range(m):
			#get the current sample
			X_i = X_mat[i, :]
			y_i = y[i]
			y_hat = X_i.dot(theta)
			#predict the label of the ith-example
			if y_hat * y_i <= 0: 	# the prediction is wrong
				theta += y_i*X_i 	# we update the theta
				num_errors = num_errors + 1
	return theta, num_epochs


def perceptron_np_predict(Xdata, theta):	# prediction on pts in Xdata
	y_pred = Xdata.dot(theta)
	y_pred[y_pred>0] = 1
	y_pred[y_pred<=0] = -1
	return y_pred 


def perceptron_np_acc(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def perceptron_np_homemade(X, y, max_epochs):
	m, n = X.shape				# number of features in the training set
	theta = np.zeros(n + 1)		# we consider the +1 since we need to insert the bias as well
	X_mat = np.hstack((np.ones((m, 1)), X))

	# train
	theta, num_epochs = perceptron_np_train(X_mat, y, theta, max_epochs)
	
	# prediction and accuracy
	y_pred = perceptron_np_predict(X_mat, theta)

	# accuracy on prediction
	acc = perceptron_np_acc(y,y_pred)

	return theta, num_epochs, y_pred, acc



### OUR EXAMPLE
# let's build a custom dataset
X, y = datasets.make_blobs(n_samples=150,n_features=2,
                           centers=2,cluster_std=1.05,
                           random_state=2)                        
y[y==0] = -1
# Plotting data
fig = plt.figure(figsize=(7,5))
plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.title('Random Classification Data with 2 classes')


theta, num_epochs, y_pred, acc = perceptron_np_homemade(X, y, 30)
print(theta, '\n', num_epochs, '\n', y_pred, '\n', acc)

# Plotting
fig = plt.figure(figsize=(7,5))
plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.title('Random Classification Data with 2 classes')

# ADD THE DECISION BOUNDARY
# The Line is y=mx+c
x1 = np.array([min(X[:,0]), max(X[:,0])])
m = -theta[1]/theta[2]
c = -theta[0]/theta[2]
x2 = m * x1 + c
plt.plot(x1, x2, 'y-')