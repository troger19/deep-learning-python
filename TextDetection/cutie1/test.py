
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"d:\Java\Tesseract\tesseract.exe"
)

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import DBSCAN
import random
import cv2
from PIL import Image
from glob import glob


# In[11]:


# Get all file names 
receipt_files = glob('./large-receipt-image-dataset-SRD/*')
# print(receipt_files[:3]) # print first 3 files



receipt_0 = Image.open(receipt_files[0])

print(pytesseract.image_to_string(receipt_0,lang='SLK'))
plt.figure(figsize=(10, 10))
plt.imshow(receipt_0)
plt.axis('off')
plt.show()

#  get characters and correspongding position
ocr_out = pytesseract.image_to_boxes(receipt_0)

#  plot position of each character detectedon the image
img = cv2.imread(receipt_files[0])
h, w, _ = img.shape
for bbox in ocr_out.split('\n'):
    split = bbox.split(' ')
    text = split[0]
    if (text == ''):
        break
    point1 = (int(split[1]), h - int(split[2]))
    point2 = (int(split[3]), h - int(split[4]))
    img = cv2.rectangle(img, point1, point2, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(img)
plt.axis('off')



fig = plt.figure(figsize=(15,10))

plt.tight_layout()
plt.subplots_adjust(top=0.95, wspace=0.1, hspace=0)

points = []
for bbox in ocr_out.split('\n'):
    split = bbox.split(' ')
    text = split[0]
    if (text == ''):
        break
    point1 = (int(split[3]) + int(split[1])) / 2
    point2 =  (h -  int(split[4]) + h - int(split[2]))/2
    points.append([point1,point2])

img = cv2.imread(receipt_files[0])
white = np.zeros((h,w))
for point in points:
    x = int(point[0])
    y = int(point[1])
    img = cv2.circle(img, (x,y) , 3, (255,0,0))
    white = cv2.circle(white, (x,y) , 5, (255,255,255), 6)


plt.subplot(1,2,1)
#plt.title()
plt.imshow(img)
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(white)
plt.axis('off')

plt.show()


# <div class="alert alert-success">
#     On the right, you'll find the position of each character found by Tesseract OCR. The location of a single character is not useful. On the right we can see that character belonging to the same word or paragraph are closed. This kind of problem can be tackled with clustering algorithms.
#    <br> <br> The following tries to gather points that correspond to characters belonging to the same word. </div>

# In[18]:


clustering = DBSCAN(eps=50, min_samples=2).fit_predict(points)


# <div class="alert alert-success">
#     One way to gather points is to use clustering algorithms. I chose DBSCAN because one don't have to precise the number of cluster and it deals well with outliers. In our case we can see above that we have several outsiders and that it would be nice to remove them. <br><br> Only <b>eps</b> parameter need to be set which is the minimal distance between two points to be considered as neighbor and then belong to the same cluster.
# <br><br>
# The value of 27 appears to work well on this image after several tries.</div>

# In[19]:


# Just for nice plotting: get a color for each cluster
dico_color = {}
for i in range(np.max(clustering)+1):
    dico_color[i] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))


# Plot clusters
plt.figure(figsize=(12,16))
img = cv2.imread(receipt_files[0])
h, w, _ = img.shape

# plot a rectangle around each character with the color of the cluster
for en,bbox in enumerate(ocr_out.split('\n')):
    split = bbox.split(' ')
    text = split[0]
    if (text == ''):
        break
    bottom_left = (int(split[1]), h - int(split[2]))
    top_right = (int(split[3]), h -  int(split[4]))
    if clustering[en] != -1:
        img = cv2.rectangle(img, bottom_left, top_right, dico_color[clustering[en]], 2)

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')

#  plot a rectangle around each cluster
img = cv2.imread(receipt_files[0])
dico_area_location = {} #  store location of each cluster for the following
for i in range(np.max(clustering)+1):
    wh = np.where(clustering==i) #  select points from cluster i
    arr = np.array(ocr_out.split('\n'))[wh]
    LX,LY = [], []
    for en,bbox in enumerate(arr):
        split = bbox.split(' ')
        LX.append(int(split[1]))
        LX.append(int(split[3]))
        LY.append( h - int(split[2]))
        LY.append( h - int(split[4]))
    bottom_left = (min(LX)-5,min(LY)-5) # among every character in the cluster, take min(x), min(y)
    top_right = (max(LX)+5,max(LY)+5) # among every character in the cluster, take max(x), max(y)
    dico_area_location[i] = [bottom_left, top_right]

    img = cv2.rectangle(img, bottom_left, top_right, dico_color[i], 2) # draw a rectangle containing
    #  every character in the cluster

plt.subplot(1,2,2)
plt.imshow(img)
plt.axis('off')
plt.show()

