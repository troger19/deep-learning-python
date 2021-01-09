import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from IPython.display import Image
# %matplotlib inline

## print number of GPU.. not necessary but faster
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# obtain a copy of a single pretrained MobileNet with weights that were saved from being trained on ImageNet images
#its downloaded only once
mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'images\\'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# load the image
# Image(filename='lizard.jpg', width=300,height=200)

# test lizard
lizard_image = prepare_image('lizard.jpg')
predictions = mobile.predict(lizard_image)

results_lizard = imagenet_utils.decode_predictions(predictions)
print(results_lizard)

# test espresso
espresso_image = prepare_image('espresso.jpg')
predictions = mobile.predict(espresso_image)

results_espresso = imagenet_utils.decode_predictions(predictions)
print(results_espresso)

# test strawberry
strawberry_image = prepare_image('strawberry.jpg')
predictions = mobile.predict(strawberry_image)

results_strawberry = imagenet_utils.decode_predictions(predictions)
print(results_strawberry)

# test beach.jpg
beach_image = prepare_image('beach.jpg')
predictions = mobile.predict(beach_image)

results_beach = imagenet_utils.decode_predictions(predictions)
print(results_beach)

# Yamaha.jpg
yamaha_image = prepare_image('Yamaha.jpg')
predictions = mobile.predict(yamaha_image)

results_yamaha = imagenet_utils.decode_predictions(predictions)
print(results_yamaha)