import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras

IMG_SIZE = (64,64)

# It can be used to reconstruct the model identically.
saved_model = keras.models.load_model("../Classification/2/saved_model/1")

# Check for a single image
test_image=cv2.resize(cv2.imread(os.path.join('test/obr1.jpg')),tuple(reversed(IMG_SIZE)))
test_image = np.expand_dims(test_image, axis = 0)
result = saved_model.predict(test_image)
print(f'Percentage for each class:{result} \nmodel prediction :{np.argmax(result, axis=1)}')

# Check for a single image
test_image=cv2.resize(cv2.imread(os.path.join('test/obr2.jpg')),tuple(reversed(IMG_SIZE)))
test_image = np.expand_dims(test_image, axis = 0)
result = saved_model.predict(test_image)
print(f'Percentage for each class:{result} \nmodel prediction :{np.argmax(result, axis=1)}')

# store picture to text file as array input for prediction
# import json
# img1Text = test_image.tolist()
# with open('data.json', 'w') as outfile:
#     json.dump(img1Text, outfile)