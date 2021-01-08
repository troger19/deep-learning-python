import cv2
import numpy as np
import pytesseract
import os


# pytesseract.pytesseract.tesseract_cmd = 'd:\\Java\\Tesseract\\tesseract.exe'

per = 20
# path = 'img_crop'
path = '..\\Datasets\\faktury\\pdf\\orange\\'
imgQ = cv2.imread(path + '\\model_image.jpg')
# imgQ = cv2.imread('..\\Datasets\\faktury\\compressed_photo\\straight\\orange\\1.jpg')
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//3,h//3))
# cv2.imshow('Output',imgQ)
# cv2.waitKey(0)

#
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
# impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
# cv2.imshow('Output',impKp1)
# cv2.waitKey(0)
#
myPicList = os.listdir(path)
print(myPicList)
# for j,y in enumerate(myPicList):
img1 = cv2.imread('..\\Datasets\\faktury\\compressed_photo\\cropped\\orange\\4.jpg')
# img1 = cv2.imread('..\\Datasets\\faktury\\pdf\\orange\\2_rotate.jpg')
#     # img_photo = cv2.resize(img_photo, (w // 3, h // 3))
#     cv2.imshow(y, img_photo)
kp2, des2 = orb.detectAndCompute(img1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des2,des1)
matches.sort(key= lambda x: x.distance)
good = matches[:int(len(matches) * (per / 100))]
imgMatch = cv2.drawMatches(img1, kp2, imgQ, kp1, good[:100], None, flags=2)
imgMatch = cv2.resize(imgMatch, (w // 2, h // 2))
# #
cv2.imshow('matched', imgMatch)
# #
srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
imgScan = cv2.warpPerspective(img1, M, (w, h))

## orange amount ROI
cv2.rectangle(imgScan, (1432, 990), (1546, 1042), (0, 255, 0), thickness=2)
## o2 amount ROI
# cv2.rectangle(imgScan, (1400, 954), (1564, 998), (0, 255, 0), thickness=2)
# img_extracted_field = imgScan[990:1042, 1432:1546]
# extracted_field = pytesseract.image_to_string(img_extracted_field, config='--psm 7 --oem 3', lang='SLK').strip()
# print(extracted_field)


imgScan = cv2.resize(imgScan, (w // 2, h // 2))
cv2.imshow('1.png', imgScan)
# cv2.imshow("Output",imgQ)
cv2.waitKey(0)