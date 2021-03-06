# -*- coding: utf-8 -*-
"""fashion_mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UdozdIDOajrjUEvTiWIrXAXIH8HYIQAM
"""

import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape,y_train.shape)
print(len(x_train))
print(len(x_test))

#labels,labels look at me now lol
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

plt.imshow(x_test[1])

x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32')/255

x_train, x_valid = x_train[5000:],x_train[:5000]  #i just splitted training set 
y_train, y_valid = y_train[5000:],y_train[:5000]  #for cross validation

w,h = 28,28
x_train = x_train.reshape(x_train.shape[0],w,h,1)
x_test  = x_test.reshape(x_test.shape[0],w,h,1)
x_valid = x_valid.reshape(x_valid.shape[0],w,h,1)

y_train = tf.keras.utils.to_categorical(y_train) #this turns number into last layer output
y_test  = tf.keras.utils.to_categorical(y_test)
y_valid = tf.keras.utils.to_categorical(y_valid)


print(x_train.shape[0],'train')
print(x_test.shape[0],'test')
print(x_valid.shape[0],'validation')

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=2,padding='same',
                                 activation='relu',input_shape=(w,h,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', 
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

#s=model.predict(x_train[0].reshape(1,w,h,1))  for observing images jus after convolution
#s= s.reshape(28,28,64)
#plt.imshow(s[:,:,45])

checkpoint = ModelCheckpoint(filepath='model.weights.last.hdf5',verbose=1,save_best_only=True)

model.fit(x_train,
          y_train,
          batch_size=64,
          epochs =10,
          validation_data = (x_valid,y_valid),
          callbacks=[checkpoint])

# evaluating time!

test_idx = 23
examp = x_test[test_idx]
plt.imshow(examp.reshape(28,28))
s=model.predict(examp.reshape(1,w,h,1))

res= np.argmax(s)
print("activated val:",s[:,res])
print(res)
print(fashion_mnist_labels[res])
score = model.evaluate(x_test, y_test, verbose=0)