import tensorflow as tf
import time

with tf.device('/cpu:0'):
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), include_top=False)

model.summary()


#estimate the time for compiling the model
t = time.time()
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
print("Compile time: ", time.time() - t)

#run the model with an example

import numpy as np

img = np.random.rand(1,224,224,3)

with tf.device('/gpu:0'):
    start = time.time()
    model.predict(img)
    end = time.time()
    print("GPU time: ", end-start)

with tf.device('/cpu:0'):
    start = time.time()
    model.predict(img)
    end = time.time()
    print("CPU time: ", end-start)
#take the layers of the model and divide them into groups of 4

layers = model.layers

groups = []

for i in range(0, len(layers), 4):
    groups.append(layers[i:i+4])

#take the layers of the models and divide them into single groups

layers = model.layers

groups = []

for i in range(0, len(layers), 1):
    groups.append(layers[i:i+1])

#now calculate the time for each group

import numpy as np

img = np.random.rand(1,224,224,3)

for group in groups:
    with tf.device('/gpu:0'):
        start = time.time()
        model.predict(img)
        end = time.time()
        print("GPU time: ", end-start)

for group in groups:
    with tf.device('/cpu:0'):
        start = time.time()
        model.predict(img)
        end = time.time()
        print("CPU time: ", end-start)
    
#there are multiple runtimes like cpu gpu raspeberry pi, write a python program in order to compare the runtimes of the different models


