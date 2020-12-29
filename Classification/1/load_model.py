import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy

saved_model = keras.models.load_model("saved_model/1")

img=image.load_img('bikes/testing/obr1.png',target_size=(200,200))
X=image.img_to_array(img)
X = numpy.expand_dims(X,axis=0)
images = numpy.vstack([X])
val=saved_model.predict(images)
if val == 0:
    print(("bicycle"))
else:
   print("motorbike")


# store picture to text file as array input for prediction
# import json
# img1Text = images.tolist()
# with open('data.json', 'w') as outfile:
#     json.dump(img1Text, outfile)