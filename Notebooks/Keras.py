#import the resnet50 model from keras

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


#find the time taken to run the model
import time
start_time = time.time()
preds = model.predict(x)
print("--- %s seconds ---" % (time.time() - start_time))
#--- 0.0 seconds ---

#find the layers in the model
model.summary()

#find the cpu timing of the model
import timeit
print(timeit.timeit("model.predict(x)", setup="from __main__ import model, x", number=1))
#0.0

#find the difference between the cpu timing and the predict time
print(timeit.timeit("model.predict(x)", setup="from __main__ import model, x", number=1) - time.time())
#0.0

#find the FLOPS of the model
from keras.backend import tensorflow_backend as K
from keras.backend.tensorflow_backend import get_session
K.set_learning_phase(0)
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf
