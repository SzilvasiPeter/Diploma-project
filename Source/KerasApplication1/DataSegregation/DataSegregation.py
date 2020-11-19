import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Keras specific
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def my_train_test_split():
    dataframe = pd.read_csv('c:\\Users\\z0042fkb\\Documents\\GitHub\\Diploma-project\\Source\\KerasApplication1\\DataSegregation\\student_scores.csv')
    print(dataframe.shape)
    print(dataframe.head())

    ax = dataframe.plot.scatter(x='Hours', y='Scores', title='Diák eredmények', color='b')
    ax.set_xlabel("Óra")
    ax.set_ylabel("Eredmény")

    train_dataframe = dataframe.sample(frac=0.8, random_state=3)
    #train_dataframe = dataframe.sample(frac=0.8, random_state=3)
    #train_dataframe = dataframe.sample(frac=0.8, random_state=6)
    test_dataframe = dataframe.drop(train_dataframe.index)

    print('Using %d samples for training and %d for validation' % (len(train_dataframe), len(test_dataframe)))

    target_column = ['Hours']
    predictors = list(set(list(dataframe.columns))-set(target_column))
    dataframe[predictors] = dataframe[predictors]
    
    print(dataframe.describe())

    X = dataframe[predictors].values
    y = dataframe[target_column].values

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.20, random_state=7) # 1 vs 7
    print(X_train.shape)
    print(X_train.ndim)

    print(y_train.shape)
    print(y_train.ndim)

    print(X_test.shape)
    print(X_test.ndim)

    print(y_test.shape)
    print(y_test.ndim)
    #print(X_train)
    #print(y_train)
    #print(X_test)
    #print(y_test)

    #data1 = np.vstack((X_train.flatten(), X_test.flatten())).T
    data2 = np.vstack((y_train.flatten(), y_test.flatten())).T
    #print(data1.shape)
    #print(data2.shape)

    #df1 = pd.DataFrame(data1, columns=['Scores', 'Hours'])
    df2 = pd.DataFrame(data2, columns=['Scores', 'Hours'])

    #df1.plot.scatter(x='Hours', y='Scores', title='Student Score Dataset', color='b')
    ax2 = df2.plot.scatter(x='Hours', y='Scores', title='Diák eredmények', color='r')
    ax2.set_xlabel("Óra")
    ax2.set_ylabel("Eredmény")
    plt.show()
    
    model = define_model()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(X_train, X_test, epochs=20)

    pred_train = model.predict(X_train)
    train_error = np.sqrt(mean_squared_error(X_test,pred_train))
    print('Mean squared error on train: ', np.sqrt(mean_squared_error(X_test,pred_train)))
    
    pred = model.predict(y_train)
    print("pred shape: ", pred.shape)
    print(pred)
    test_error = np.sqrt(mean_squared_error(y_test,pred))
    print('Mean squared error on test: ', np.sqrt(mean_squared_error(y_test,pred)))

    print("Error difference: ", train_error - test_error)

    #data3 = np.vstack((y_train.flatten(), pred.flatten())).T
    #df3 = pd.DataFrame(data3, columns=['Scores', 'Hours'])
    #df2.plot.scatter(x='Hours', y='Scores', title='Student Score Dataset', color='y')

    #plt.show()

def define_model():
    print("hello")
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation="relu"))
    #model.add(Dense(10, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))

    return model

if __name__ == "__main__":
	my_train_test_split()
