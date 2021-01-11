import cv2
import imutils
import pytesseract

path = 'cars\\'
image = cv2.imread(path + 'car4.jpg')


image=imutils.resize(image,width= 500)


# cv2.imshow('original image',image)
# cv2.waitKey(0)


gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray,11,17,17)

edged = cv2.Canny(gray,170,200)

cnts, new = cv2.findContours(edged.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1  =image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)

cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None


image2 = image.copy()
# cv2.drawContours(image2,cnts,-1,(0,255,0),3)
# cv2.imshow('Top 30 Countours',image2)
# cv2.waitKey(0)

count = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i,0.02*perimeter,True)
    if(len(approx) == 4):
        NumberPlateCount = approx
        x,y,w,h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]

        # gray = cv2.cvtColor(crp_img, cv2.COLOR_BGR2GRAY)
        # adaptiveThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)

        # gray_image = cv2.cvtColor(crp_img, cv2.COLOR_BGR2GRAY)
        # threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('crp_img image', crp_img)
        cv2.waitKey(0)
        text = pytesseract.image_to_string(crp_img,lang='eng',config="--psm 7").strip()
        print(text)

        # cv2.imwrite(str(name) + '.png',crp_img)
        # name +=1

        break

# cv2.drawContours(image,[NumberPlateCount],-1,(0,255,0),3)
# cv2.imshow('Final image',image)
# cv2.waitKey(0)

