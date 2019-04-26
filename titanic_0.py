from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt
 
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation, Dense

#print(tf.__version__)

#load the data and take a peek
train_disaster = pd.read_csv("titanicTrain.csv", sep = ",")
pred_disaster = pd.read_csv("titanicTest.csv", sep = ",")
#print(disaster.head())

#subset for only the features that are likely to be significant
train_disaster = train_disaster[['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].dropna()
pred_disaster = pred_disaster[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].dropna()

#Ref: http://jamilgafur.com/wp-content/uploads/2017/09/Tensorflowtitanictutorial.html

#process the data
#pandas doesn't have an easy way to convert categorical data into numerically represented factors
#so we do it (psuedo) manually for the two categorical features we have
def preprocess_gender(df):
	gender = LabelEncoder()
	gender.fit(['male', 'female'])
	df['Sex'] = gender.transform(df['Sex'])

def preprocess_embark(df):
	emb = LabelEncoder()
	emb.fit(['S', 'C','Q'])
	df["Embarked"] = emb.transform(df["Embarked"])

preprocess_gender(train_disaster)
preprocess_embark(train_disaster)

preprocess_gender(pred_disaster)
preprocess_embark(pred_disaster)


print(train_disaster.head())
#print(pred_disaster.head())
#print(train_disaster.shape)


#define the model


# create model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#figuring out which keras setup gives me a binary outcome
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
train_d_features = train_disaster[[ 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train_d_outcome = train_disaster[['Survived']]
model.fit(train_d_features, train_d_outcome, epochs=5)

#test the model
pred_d_features = pred_disaster[[ 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
preds = model.predict(pred_d_features)
print(preds[0])



