import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras


def my_train_test_split():
    dataframe = pd.read_csv("c:\\Users\\z0042fkb\\Documents\\GitHub\\Diploma-project\\Source\\KerasApplication1\\DataSegregation\\student_scores.csv")
    print(dataframe.shape)
    print(dataframe.head())
    train_dataframe = dataframe.sample(frac=0.8, random_state=1)
    #train_dataframe = dataframe.sample(frac=0.8, random_state=3)
    #train_dataframe = dataframe.sample(frac=0.8, random_state=6)
    test_dataframe = dataframe.drop(train_dataframe.index)

    print("Using %d samples for training and %d for validation" % (len(train_dataframe), len(test_dataframe)))

    target_column = ['Hours']
    predictors = list(set(list(dataframe.columns))-set(target_column))
    dataframe[predictors] = dataframe[predictors]
    
    print(dataframe.describe())

    X = dataframe[predictors].values
    y = dataframe[target_column].values

    train, test = train_test_split(dataframe, test_size=0.20, random_state=1)
    print(X_train.shape); print(X_test.shape)

    train_dataframe.plot.scatter(x='Hours', y='Scores', title='Student Score Dataset', color="b")
    test_dataframe.plot.scatter(x='Hours', y='Scores', title='Student Score Dataset', color="r")

    plt.show()

    model = define_model()

    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    model.fit(train, y_train, epochs=20)

    pred_train= model.predict(X_train)
    print(np.sqrt(mean_squared_error(y_train,pred_train)))
    
    pred= model.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test,pred)))
    # create a figure and axis
    #fig, ax = plt.subplots()

    # scatter the sepal_length against the sepal_width
    #ax.scatter(dataframe['Hours'], dataframe['Scores'])
    # set a title and labels
    #ax.set_title('Student Dataset')
    #ax.set_xlabel('Hours')
    #ax.set_ylabel('Scores')

def define_model():
    print("hello")
    model = Sequential()
    model.add(Dense(30, input_dim=1, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(1))

    return model

if __name__ == "__main__":
	my_train_test_split()
