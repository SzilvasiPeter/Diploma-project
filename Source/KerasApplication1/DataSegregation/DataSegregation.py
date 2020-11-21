import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def data_segregation():
    X, y = load_data()
    # X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.20, random_state=7) # 1 or 7 randomstate
    # visualize_datasets(X_train, y_train, X_test, y_test)

    fold_number = 4
    kfold_data_split(X, y, fold_number)

    #model = define_model()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    #model.fit(X_train, X_test, epochs=20)


def load_data():
    dataframe = pd.read_csv('student_scores.csv')
    print(dataframe.shape)
    print(dataframe.head())
  
    target_column = ['Hours']
    predictors = list(set(list(dataframe.columns))-set(target_column))

    dataframe[predictors] = dataframe[predictors]
    print(dataframe.describe())

    X = dataframe[predictors].values
    y = dataframe[target_column].values

    return X, y


def kfold_data_split(X, y, fold_number=4):
    KFCrossValidator = KFold(n_splits=fold_number, shuffle=False)
    KFdataset = KFCrossValidator.split(X)
    
    train_test_error_diff = 0
    for train_index, test_index in KFdataset:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, y_train = X[train_index], X[test_index]
        X_test, y_test = y[train_index], y[test_index]

        model = define_model()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        model.fit(X_train, X_test, epochs=20)

        train_test_error_diff += evaluate_model(model, X_train, y_train, X_test, y_test)

    print('Average train test error diff: ', train_test_error_diff / fold_number)


def visualize_datasets(X_train, y_train, X_test, y_test):
    train_dataset = np.vstack((X_train.flatten(), X_test.flatten())).T
    test_dataset = np.vstack((y_train.flatten(), y_test.flatten())).T
  
    train_dataframe = pd.DataFrame(train_dataset, columns=['Scores', 'Hours'])
    test_dataframe = pd.DataFrame(test_dataset, columns=['Scores', 'Hours'])
  
    ax_train = train_dataframe.plot.scatter(x='Hours', y='Scores', title='Diák eredmények', color='b')
    ax_train.set_xlabel("Óra")
    ax_train.set_ylabel("Eredmény")

    ax_test = test_dataframe.plot.scatter(x='Hours', y='Scores', title='Diák eredmények', color='r')
    ax_test.set_xlabel("Óra")
    ax_test.set_ylabel("Eredmény")

    plt.show()


def define_model():
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation="relu"))
    #model.add(Dense(10, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    train_error = np.sqrt(mean_squared_error(X_test, pred_train))
    print('Mean squared error on train: ', np.sqrt(mean_squared_error(X_test, pred_train)))
  
    pred = model.predict(y_train)
    test_error = np.sqrt(mean_squared_error(y_test, pred))
    print('Mean squared error on test: ', np.sqrt(mean_squared_error(y_test, pred)))
  
    print("Error difference: ", train_error - test_error)

    return train_error - test_error


if __name__ == "__main__":
	data_segregation()
