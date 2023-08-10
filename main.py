import tensorflow as tf #biblioteca de Deep Learning
import matplotlib.pyplot as plt
import numpy as np

import os

import cv2 as cv

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape #Aqui em baixo é possivel ver o tamanho do dataset, juntamente com a largura e o comprimento.

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #Adicionando a camada de entrada

model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu)) #rectified linear unit
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('numeros.model')

img = cv.imread('/home/files/img1.png')[:,:,0]
img = np.array([img])

result = model.predict(img)
print(f'O número identificado foi {np.argmax(result)}')