import cv2
import pytesseract

img = cv2.imread('..\\Datasets\\faktury\\pdf\\orange\\1.png')
# pytessearct only expects RGB but opencv is in BGR
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# target_text = 'Suma na úhradu'
target_text = 'suma'

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## turn the pixel white which have values between 100–255
threshold_img = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)[1]
# threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# threshold_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# dst = cv2.resize(threshold_img, (int(width / 2), int(height / 2)))
dst = cv2.resize(threshold_img, (int(width / 2), int(height / 2)))

cv2.imshow('threshold_img',dst)
cv2.waitKey(0)

# ## Detecting words
hImg,wImg,_ = img.shape
boxes = pytesseract.image_to_data(threshold_img, lang='SLK')
print(boxes)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        # print(b)
        if len(b) == 12:
            if target_text in b[11].lower():
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),3)
            # cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)

dst = cv2.resize(img, (int(width / 2), int(height / 2)))
RGB_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
cv2.imshow('Result',RGB_img)
#wait for infinity
cv2.waitKey(0)