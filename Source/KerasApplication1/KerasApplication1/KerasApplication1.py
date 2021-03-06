﻿import tensorflow as tf
import tensorflow
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

def heart_disease_classification():
	train_ds, val_ds = prepare_dataset()
	model = build_model(train_ds, val_ds)
	model.fit(train_ds, epochs=50, validation_data=val_ds)
	predict_sample(model)


def prepare_dataset():
	file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
	dataframe = pd.read_csv(file_url)
	print(dataframe.shape)
	print(dataframe.head())
	
	val_dataframe = dataframe.sample(frac=0.2)
	train_dataframe = dataframe.drop(val_dataframe.index)
	
	print("Using %d samples for training and %d for validation" % (len(train_dataframe), len(val_dataframe)))
	
	train_ds = dataframe_to_dataset(train_dataframe)
	val_ds = dataframe_to_dataset(val_dataframe)
	
	for x, y in train_ds.take(1):
		print("Input:", x)
		print("Target:", y)
	
	train_ds = train_ds.batch(32)
	val_ds = val_ds.batch(32)

	return train_ds, val_ds


def build_model(train_ds, val_ds):
	sex = keras.Input(shape=(1,), name="sex", dtype="int64")
	cp = keras.Input(shape=(1,), name="cp", dtype="int64")
	fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
	restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
	exang = keras.Input(shape=(1,), name="exang", dtype="int64")
	ca = keras.Input(shape=(1,), name="ca", dtype="int64")

	thal = keras.Input(shape=(1,), name="thal", dtype="string")

	age = keras.Input(shape=(1,), name="age")
	trestbps = keras.Input(shape=(1,), name="trestbps")
	chol = keras.Input(shape=(1,), name="chol")
	thalach = keras.Input(shape=(1,), name="thalach")
	oldpeak = keras.Input(shape=(1,), name="oldpeak")
	slope = keras.Input(shape=(1,), name="slope")

	all_inputs = [
		sex,
		cp,
		fbs,
		restecg,
		exang,
		ca,
		thal,
		age,
		trestbps,
		chol,
		thalach,
		oldpeak,
		slope,
	]

	sex_encoded = encode_integer_categorical_feature(sex, "sex", train_ds)
	cp_encoded = encode_integer_categorical_feature(cp, "cp", train_ds)
	fbs_encoded = encode_integer_categorical_feature(fbs, "fbs", train_ds)
	restecg_encoded = encode_integer_categorical_feature(restecg, "restecg", train_ds)
	exang_encoded = encode_integer_categorical_feature(exang, "exang", train_ds)
	ca_encoded = encode_integer_categorical_feature(ca, "ca", train_ds)

	thal_encoded = encode_string_categorical_feature(thal, "thal", train_ds)

	age_encoded = encode_numerical_feature(age, "age", train_ds)
	trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
	chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
	thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
	oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
	slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

	all_features = layers.concatenate(
		[
			sex_encoded,
			cp_encoded,
			fbs_encoded,
			restecg_encoded,
			exang_encoded,
			slope_encoded,
			ca_encoded,
			thal_encoded,
			age_encoded,
			trestbps_encoded,
			chol_encoded,
			thalach_encoded,
			oldpeak_encoded,
		]
	)
	x = layers.Dense(32, activation="relu")(all_features)
	x = layers.Dropout(0.5)(x)
	output = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(all_inputs, output)
	model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
	return model

def dataframe_to_dataset(dataframe):
	dataframe = dataframe.copy()
	labels = dataframe.pop("target")
	dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
	dataset = dataset.shuffle(buffer_size=len(dataframe))
	return dataset


def encode_numerical_feature(feature, name, dataset):
	normalizer = Normalization()

	feature_ds = dataset.map(lambda x, y: x[name])
	feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

	normalizer.adapt(feature_ds)

	encode_feature = normalizer(feature)
	return encode_feature


def encode_string_categorical_feature(feature, name, dataset):
    index = StringLookup()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    index.adapt(feature_ds)

    encoded_feature = index(feature)

    encoder = CategoryEncoding(output_mode="binary")

    feature_ds = feature_ds.map(index)

    encoder.adapt(feature_ds)

    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    encoder = CategoryEncoding(output_mode="binary")

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    encoder.adapt(feature_ds)

    encoded_feature = encoder(feature)
    return encoded_feature


def predict_sample(model):
	sample = {
		"age": 60,
		"sex": 1,
		"cp": 1,
		"trestbps": 145,
		"chol": 233,
		"fbs": 1,
		"restecg": 2,
		"thalach": 150,
		"exang": 0,
		"oldpeak": 2.3,
		"slope": 3,
		"ca": 0,
		"thal": "fixed",
	}

	input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
	predictions = model.predict(input_dict)

	print("This particular patient had a %.1f percent probability of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],))

if __name__ == "__main__":
	heart_disease_classification()