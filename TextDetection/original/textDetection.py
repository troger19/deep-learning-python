import cv2
import pytesseract

# specify the path to tesseract or install module to Python environemnt
# pytesseract.pytesseract.tesseract_cmd = 'd:\\Java\\Tesseract\\tesseract.exe'

img = cv2.imread('2021-02-11.jpg')
img_photo = cv2.imread('2021-02-11.jpg')
# pytessearct only expects RGB but opencv is in BGR
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# image to text
print(pytesseract.image_to_string(img,lang='SLK'))

### Detecting characters
##extract width and height of the image. 3. parmeter = 3 which means color
hImg,wImg,_ = img_photo.shape
boxes = pytesseract.image_to_boxes(img_photo, lang='SLK',config='--psm 10')
# boxes = pytesseract.image_to_boxes(img_photo, lang='SLK',config='--oem 2 --psm 10 -c tessedit_char_whitelist=0123456789')
for b in boxes.splitlines():
    b=b.split(' ')
    print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img_photo,(x,hImg-y),(w,hImg-h),(0,0,255),1)
    cv2.putText(img_photo,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)

cv2.imshow('Result',img_photo)
#wait for infinity
cv2.waitKey(0)

## Detecting words
hImg,wImg,_ = img.shape
boxes = pytesseract.image_to_data(img, lang='SLK')
print(boxes)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        print(b)
        if len(b) ==12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),3)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)

cv2.imshow('Result',img)
#wait for infinity
cv2.waitKey(0)