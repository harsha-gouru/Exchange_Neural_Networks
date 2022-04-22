#import the tensorflow model from the library
#compile the model
#find the time difference between the estimate time and the actual time for compiling the model
#find the time difference between the estimate time and the actual time for running the model\
#find the time difference between the estimate time and the actual time for saving the model
#find the time difference between the estimate time and the actual time for loading the model
#find the time difference between the estimate time and the actual time for predicting the model

import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

NAME = "Cats-vs-Dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

#start the time
t= time.time()

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])

model.save("64x2.model")

loaded_model = tf.keras.models.load_model("64x2.model")

loaded_model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

score = loaded_model.evaluate(X, y, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

for i in range(len(predictions)):
    prediction = loaded_model.predict(X[i].reshape(1, 64, 64, 3))
    print("Prediction: ", prediction)
    print("Actual: ", y[i])
    print()

#end the time
t2 = time.time()

#time taken to compile the model
t3 = t2 - t
