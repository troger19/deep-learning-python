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
# pdf = convert_from_path('..\\Datasets\\faktury\\orange\\1.pdf')
from CustomOCR.Utils import find_template, extract_fields

pdf = convert_from_path('..\\Datasets\\faktury\\orange\\5.pdf')

# Save PDF as Image so we can select the ROI
# pdf[0].save('image.jpg', 'JPEG')

# find the page with QR code
qr_code_page_number = -1
for pageNumber, page in enumerate(pdf):
    extracted_page = decode(page)
    if (len(extracted_page) == 1):  # nasla prave 1 QR kod
        print(extracted_page[0].data.decode('utf-8'))
        qr_code_page_number = pageNumber
        break

print(qr_code_page_number)

# extract data
data = pytesseract.image_to_string(pdf[qr_code_page_number], lang='SLK')
print(data)

# try to find template for the image
template = find_template(data)

# if template is found extract based on ROI, if not try to find the keywords and guess the ROI
if template:
    extracted_fields = extract_fields(pdf[qr_code_page_number],template['ROI'])
    print(extracted_fields)
else:
    print('no')
