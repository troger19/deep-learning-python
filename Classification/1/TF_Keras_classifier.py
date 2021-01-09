
# coding: utf-8

# ## Mountain_bike / Road_Bike Classifier
# ### Author : Shreyas Vaidyanath

# In[1]:


#importing dependencies
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras

IMAGE_CATEGORY = 'bikes' #shoes, people

# In[2]:


# Setting up constants
IMG_SIZE = (64,64)
EPOCHS =20


# In[4]:


# Loading the dataset
# x - images , y - class identifier ['mountain_bike', 'road_bike'] 
x, y = [], []

# Iterate through each folder in path
cls_count = 0
for sub_class in os.listdir('classifier_data_'+IMAGE_CATEGORY+'/training'):
    for file in os.listdir(os.path.join('classifier_data_'+IMAGE_CATEGORY+'/training', sub_class)):
        y += [cls_count]
        x += [cv2.resize(cv2.imread(os.path.join('classifier_data_'+IMAGE_CATEGORY+'/training', sub_class, file)),tuple(reversed(IMG_SIZE)))]
    cls_count += 1

# Convert to numpy array
x = np.array(x)
y = np.eye(cls_count)[y]  # One hot encode


# In[5]:


# Initializing the model
model = keras.Sequential([
    keras.layers.Conv2D(32,3, input_shape = (64, 64, 3), activation = 'relu'),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Conv2D(32,3, activation = 'relu'),
    keras.layers.MaxPool2D(strides=2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])


# In[6]:


# Initializing the adam optimizer
Optimizer = keras.optimizers.Adam
Opt=Optimizer(lr=1e-3, decay=1e-3 / EPOCHS)


# In[7]:


# Compiling the model
# after trial and error with different loss functions, this one had the optimal result
model.compile(optimizer=Opt, 
              loss ='categorical_crossentropy',
              metrics=['accuracy'])


# In[8]:


model.fit(x,y,epochs=EPOCHS)
print('End of Training')


# In[9]:


# iterate through selected test images
x_test, filename = [], []
# Iterate through each image-file in path
for file in os.listdir('classifier_data_'+IMAGE_CATEGORY+'/test'):
        filename += [file]
        x_test += [cv2.resize(cv2.imread(os.path.join('classifier_data_'+IMAGE_CATEGORY+'/test', file)),tuple(reversed(IMG_SIZE)))]

# Convert to numpy array
x_test = np.array(x_test)


# In[10]:


# predicting resuls
predictions = model.predict(x_test)
predicted_cls = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)

# In[11]:


#printing class predictions

# In[12]:


#displaying as a pandas df where 0:mountain_bike,1:road_bike
import pandas as pd
pd.DataFrame({'Actual':filename,
 'Predicted_class':predicted_cls,
 'confidence':confidence
})


# In[13]:


# Check for a single image
# test_image=cv2.resize(cv2.imread(os.path.join('classifier_data_bikes/test/obr1.jpg')),tuple(reversed(IMG_SIZE)))
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# print(f'Percentage for each class:{result} \nmodel prediction :{np.argmax(result, axis=1)}')


# In[14]:


# Check for a single image
# test_image=cv2.resize(cv2.imread(os.path.join('classifier_data_bikes/test/motorbike6.jpg')),tuple(reversed(IMG_SIZE)))
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# print(f'Percentage for each class:{result} \nmodel prediction :{np.argmax(result, axis=1)}')


# In[15]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


conf = np.around(confidence,4)


# In[19]:


fig = plt.figure(figsize=(16,16))
for i in range(6):
    # print(cnt,data)
    img=fig.add_subplot(6,5,i+1)
    if predicted_cls[i] ==1:
        str_label ='Motorbike'
    else:
        str_label = 'Bike'
    img.imshow(x_test[i])
    img.text(30,5,f'confidence:{conf[i]}', color ='red')
    plt.title(str_label)
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
plt.show()

#save model
model.save('saved_model/1')