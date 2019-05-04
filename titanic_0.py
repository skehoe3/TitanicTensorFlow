#import necessary libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation, Dense


#load the data
train_disaster = pd.read_csv("titanicTrain.csv", sep = ",")
pred_disaster = pd.read_csv("titanicTest.csv", sep = ",")

#subset the original data for only the features that have meaning
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


# create model
model = Sequential()
#add a layer and specify the number of features; the dimension of the data frame minus the column to predict on.
#relu is the activitation function; it is considered to be the best activation function the majority of the time.
model.add(Dense(12, input_dim=7, activation='relu'))  
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
#optimizer changes weights and biases during the training process.
#we use binary crossentropy as our loss because this is a binary classification problem.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Declare the features
train_d_features = train_disaster[[ 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
#Declare the column to be predicted
train_d_outcome = train_disaster[['Survived']]
#train the model
model.fit(train_d_features, train_d_outcome, epochs=100, shuffle = True)

#test the model
pred_d_features = pred_disaster[[ 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
preds = model.predict(pred_d_features)
print(preds[0])



