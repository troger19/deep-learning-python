import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras

IMAGE_CATEGORY = 'bikes' #shoes, people


#read the image
# img_photo = image.load_img(IMAGE_CATEGORY+"\\training\\janci\\janci1.jpg")
#add image to plot
# plt.imshow(img_photo)
#show plot
# plt.show()

# # show 3D matrix or shape
# imread = cv2.imread(IMAGE_CATEGORY+"\\training\\janci\\janci1.jpg")
# print(imread)
# print(imread.shape)

# while RGB range from 0 : 255 and I want to have it from 0 : 1 I divide all
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(IMAGE_CATEGORY+'\\training/',target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset = train.flow_from_directory(IMAGE_CATEGORY+'\\validation/',target_size=(200,200),batch_size=3,class_mode='binary')
print(train_dataset.class_indices)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation='relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),

                                    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),

                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512,activation="relu"),

                                    tf.keras.layers.Dense(1,activation="sigmoid")
                                    ])

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])

model_fit = model.fit(train_dataset,steps_per_epoch=3, epochs=10,validation_data=validation_dataset)

dir_path = IMAGE_CATEGORY+"\\testing"

for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'/'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()

    X=image.img_to_array(img)
    X = numpy.expand_dims(X,axis=0)
    images = numpy.vstack([X])
    val=model.predict(images)
    if val == 0:
        print(("bicycle"))
    else:
        print("motorbike")

# model.save('saved_model/1')
