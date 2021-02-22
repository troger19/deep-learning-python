import pdfplumber
import os
import time
import sys
import json
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

path = '..\\..\\Datasets\\faktury\\pokus\\'

invoices_list = os.listdir(path)

def find_template(filename):
    path = 'validation'
    with open(path + '\\' + filename, encoding='utf-8') as f:
        template = json.load(f)
        return template

def extract_fields(page, rois):
    # key:value pair of fields and their OCR values
    fields = {}
    for i, roi in enumerate(rois):
        image = np.array(page)
        height, width = image.shape[:2]
        cv2.rectangle(image, (roi['x1'], roi['x2']), (roi['y1'], roi['y2']), (0, 255, 0), thickness=2)
        img_extracted_field = image[roi['x2']:roi['y2'], roi['x1']:roi['y1']]
        # converting image into gray scale image
        gray_image = cv2.cvtColor(img_extracted_field, cv2.COLOR_BGR2GRAY)
        # converting it to binary image by Thresholding
        #  this step is require if you have colored image because if you skip this part
        #  then tesseract won't able to detect text correctly and this will give incorrect result
        threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # extract fields
        extracted_field = pytesseract.image_to_string(threshold_img, config='--psm 7 --oem 3', lang='SLK').strip()
        print(extracted_field)
        # add as key:value pair
        fields.update({roi['name']: extracted_field})
        # fit to the window
        dst = cv2.resize(image, (int(width / 2), int(height / 2)))
        # cv2 use BGR images, therefore needs to  be converted to RGB otherwise it will be blue
        RGB_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        RGB_extracted_image = cv2.cvtColor(img_extracted_field, cv2.COLOR_BGR2RGB)
        cv2.imshow('Output', RGB_img)
        cv2.imshow('Croped', RGB_extracted_image)
        cv2.waitKey(0)
    return fields


template = find_template('templates.json')

for i, y in enumerate(invoices_list):
    if i ==0:
        print(y)
        try:
            with pdfplumber.open(path + y) as pdf:
                for i, page in enumerate(pdf.pages[:1]):
                    first_page = page
                    img_pdf = convert_from_path(path + y, dpi=200)
                    extracted_fields = extract_fields(np.array(img_pdf[i]), template['ROI'])
                    print(extracted_fields)

        except:
            print(sys.exc_info())
            print('Nastal problem pri spracovani PDF' + path + y)


