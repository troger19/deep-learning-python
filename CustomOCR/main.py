import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import pytesseract
import json
from Utils import *

#ROI : Region of interest.. the area which we concetrate on and try to OCR the text, f.i. amount of invoice, date of pay, customer name

# Read whole PDF and convert each page to image
# pdf = convert_from_path('test.pdf')
# pdf = convert_from_path('..\\Datasets\\faktury\\pdf\\orange\\1.pdf')
from CustomOCR.Utils import find_template, extract_fields

document = convert_from_path('..\\Datasets\\faktury\\pdf\\orange\\1.pdf')

# Save PDF as Image so we can select the ROI
# document[0].save('1.jpg', 'JPEG')

# find the page with QR code
qr_code_page_number = -1
for pageNumber, page in enumerate(document):
    extracted_page = decode(page)
    if (len(extracted_page) == 1):  # nasla prave 1 QR kod
        print(extracted_page[0].data.decode('utf-8'))
        qr_code_page_number = pageNumber
        break

print(qr_code_page_number)

# extract data
# converting image into gray scale image
image = np.array(document[qr_code_page_number])
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# converting it to binary image by Thresholding
# this step is require if you have colored image because if you skip this part
# then tesseract won't able to detect text correctly and this will give incorrect result
# threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('threshold image', threshold_img)
# Maintain output window until user presses a key

data = pytesseract.image_to_string(image, lang='SLK')
print(data)

# try to find template for the image
template = find_template(data)

# if template is found extract based on ROI, if not try to find the keywords and guess the ROI
if template:
    extracted_fields = extract_fields(image,template['ROI'])
    print(extracted_fields)
else:
    print('no')
