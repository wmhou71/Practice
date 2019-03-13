# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:35:24 2019

@author: Engineer
"""

import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library
from sklearn.model_selection import train_test_split

#change feature "species" ' data type
iris = sns.load_dataset('iris')
print(iris)
iris.loc[iris["species"]=="setosa", "species"] = 0
iris.loc[iris["species"]=="versicolor", "species"] = 1
iris.loc[iris["species"]=="virginica", "species"] = 2
iris = iris.iloc[np.random.permutation(len(iris))]

#Set data and target 
features_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = np.array(iris[features_cols].values,float)
y = np.array(iris['species'].values,float)

#Normalization
X_normalized = normalize(X,axis=0)

#Creating train,test and validation data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

#Neural network module
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.utils import np_utils

#Change the label to one hot vector
'''
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
'''
y_train=np_utils.to_categorical(y_train, num_classes=3)
y_test=np_utils.to_categorical(y_test, num_classes=3)

#Neural network modeling
model=Sequential()
model.add(Dense(1000, input_dim=4, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

train_history = model.fit(X_train, y_train, validation_data = (X_test,y_test), batch_size=20, epochs=10, verbose=1)

prediction = model.predict(X_test)
y_label = np.argmax(y_test, axis=1)
predict_label = np.argmax(prediction,axis=1)

accuracy = np.sum(y_label==predict_label)/len(prediction) * 100 
print("Accuracy of the dataset", accuracy )

plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show() 