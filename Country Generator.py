#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


x_arr = pd.DataFrame()
y_arr = pd.DataFrame()

rows = 1
cols = 3
input_shape = (3,)


# Helper functions
def show_min_max(array, i):
  random_image = array[i]
  print("min and max value in image: ", random_image.min(), random_image.max())


def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(str(label_names[labels[i]]))
  plt.xticks([])
  plt.yticks([])
  plt.show()

def plot_loss(history):
    h = history.history
    x_lim = len(h['loss'])
    plt.figure(figsize=(8, 8))
    plt.plot(range(x_lim), h['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return

df = pd.read_csv('Cool.csv')
x_arr = df
print(df.head())
y_arr = df
    
# Add variables to keep track of image size
# x_arr = df['Continent', 'area', 'pop']
# print(x_arr)

x_arr = x_arr.iloc[:, 1:4]
print(x_arr.head())

y_arr = y_arr.iloc[:, 5:]
print (y_arr.head(20))

# Add a variable for amount of output classes

num_classes = 7

X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size = 0.03, random_state=9)

print(X_train.shape)


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
epoch = input('How many epochs?')
epochs = int(epoch)
model = Sequential()

model.add(Flatten(input_shape = input_shape)) 

model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(8, activation='softmax')) 

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs=epochs, shuffle=True)
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', scores[1])


# In[ ]:




