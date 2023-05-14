import numpy as np

# sklearn import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# keras import
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.random_seed import set_random_seed
from collections import Counter


def mlp_def_train(X_train, y_train, X_test, y_test, n_layers, activation_f):
    ## Model Setting
    # configuration options
    feature_vector_length = X_train.shape[1]
    num_classes = len(Counter(y_train).keys())

    # seed
    np.random.seed(123)
    set_random_seed(2)

    # Create the model
    model = Sequential()  # define how the "model" looks like
    if n_layers == 1:  # first and only layer
        model.add(
            Dense(
                units=num_classes, input_dim=feature_vector_length, activation="softmax"
            )
        )
    else:
        # first layer
        model.add(
            Dense(units=2, input_dim=feature_vector_length, activation=activation_f)
        )
        for i in range(2, n_layers):  # hidden layers
            model.add(Dense(units=2, activation=activation_f))
        # output layer
        model.add(Dense(units=num_classes, activation="softmax"))

    # Configure the model
    model.compile(
        loss="categorical_crossentropy",  # loss metric
        optimizer="sgd",  # optimizer
        metrics=["accuracy"],  # displayed metric
    )

    # see how the model looks like
    # model.summary()
    # print()

    ## Train
    # simple early stopping
    es = EarlyStopping(
        monitor="val_loss",  # quantity to be monitored
        mode="min",  # we look for decreasing patterns stop
        patience=5,  # number of epochs with no improvement
        verbose=1,  # print last epochs number
    )

    # categorization
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # train our model
    history = model.fit(
        X_train,
        y_train_cat,
        epochs=500,
        batch_size=16,
        verbose=0,
        validation_split=0.25,
        callbacks=[es],
    )

    ## Performance
    # see the testing performance
    test_results = model.evaluate(X_test, y_test_cat, verbose=0)

    return model, history, test_results


if __name__ == "__main__":
    # This won't be executed when we import the previous function
    # Here we are adding a trial on a random generated dataset

    ## DATASET
    np.random.seed(123)
    X, y = datasets.make_classification(
        n_samples=1500,
        n_features=2,
        random_state=123,
        n_redundant=0,
        scale=10,
        shift=10,
    )
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=123
    )
    # scale
    scaler = StandardScaler()
    scaler.fit(X_train)  # remember, always training!
    X_train_scl = scaler.transform(X_train)
    X_test_scl = scaler.transform(X_test)

    # GRID SEARCH
    nlayers = [1, 2, 3, 4]
    for nl in nlayers:
        m, h, test_res = mlp_def_train(
            X_train_scl, y_train, X_test_scl, y_test, n_layers=nl, activation_f="relu"
        )

        print(
            f"N layers:{nl}\t\tTrain ACC:{h.history['accuracy'][-1]:.8f}\t\tVal ACC:{h.history['val_accuracy'][-1]:.8f}"
        )
        print(
            f"Test results:\n\t - Loss: {test_res[0]} \n\t - Accuracy: {test_res[1]}%\n"
        )
