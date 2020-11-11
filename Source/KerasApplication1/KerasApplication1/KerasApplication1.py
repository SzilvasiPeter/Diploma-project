import tensorflow as tf
import tensorflow
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def heart_disease_classification():
	file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
	dataframe = pd.read_csv(file_url)
	print(dataframe.shape)


if __name__ == "__main__":
	heart_disease_classification()