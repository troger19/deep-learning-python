import cv2
import numpy as np
import pytesseract
import os

per = 25  # percentage of key points

## working
roi = [[(902, 278), (1250, 384), 'text', 'surname'],
       [(898, 946), (1512, 1090), 'text', 'cislo']]

# pytesseract.pytesseract.tesseract_cmd = 'd:\\Java\\Tesseract\\tesseract.exe'

path = 'img_photo'
# path = 'img_crop'
imgQ = cv2.imread('D:\\Java\\deep-learning-python\\Datasets\\ID\\id_model_image.jpg')
h,w,c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)


# myPicList = os.listdir(path)
# print(myPicList)
# for j,y in enumerate(myPicList):
img = cv2.imread('D:\\Java\\deep-learning-python\\Datasets\\ID\\jano_front.jpg')
#img_photo = cv2.resize(img_photo, (w // 3, h // 3))
# cv2.imshow(y, img_photo)
kp2, des2 = orb.detectAndCompute(img,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des2,des1)
matches.sort(key= lambda x: x.distance)
good = matches[:int(len(matches)*(per/100))]
imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)

# cv2.imshow(y, imgMatch)
srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
imgScan = cv2.warpPerspective(img,M,(w,h))
#cv2.imshow(y, imgScan)
imgShow = imgScan.copy()
imgMask = np.zeros_like(imgShow)

myData = []

for x,r in enumerate(roi):
   cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
   imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
   imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
       ##Display the regions of interest, which will be scan by OCR
   cv2.imshow(str(x), imgCrop)
            ## text extraction
   if r[2] == 'text':
      print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop,lang='SLK')))
      myData.append(pytesseract.image_to_string(imgCrop))

      cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                  cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

with open('DataOutput.csv', 'a+') as f:
   for data in myData:
       f.write((str(data)+','))
   f.write('\n')

imgShow = cv2.resize(imgShow, (w // 3, h // 3))
print(myData)
cv2.imshow("2", imgShow)
cv2.waitKey(0)
# cv2.imwrite('j',imgShow)


#cv2.imshow("KeyPointsQuery",impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
