import json
import os
import cv2
import pytesseract
import numpy as np


def find_template(data):
    path = 'templates'
    max = 0
    chosen_template = ''
    for filename in os.listdir(path):
        with open(path + '\\' + filename, encoding='utf-8') as f:
            template = json.load(f)
            print('Checking template : ' + template['name'])
            word_freq = {}

            for i, word in enumerate(template['keywords']):
                word_freq.update({word: data.count(word)})
            print(word_freq)
            sum_of_values = sum(word_freq.values())
            print(sum_of_values)
            if sum_of_values > max:
                max = sum_of_values
                chosen_template = filename
    print(chosen_template)
    if not chosen_template:
        print('Nenasla sa ziadna template, budem extrahovat naivnym sposobom')
        return None
    else:
        with open(path + '\\' + chosen_template, encoding='utf-8') as f:
            chosen_template = json.load(f)
        return chosen_template


def extract_fields(page,rois):
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
        extracted_field = pytesseract.image_to_string(threshold_img,  config='--psm 7 --oem 3',lang='SLK').strip()
        print(extracted_field)
        # add as key:value pair
        fields.update({roi['name'] :  extracted_field})
        # fit to the window
        dst = cv2.resize(image, (int(width / 2), int(height / 2)))
        # cv2 use BGR images, therefore needs to  be converted to RGB otherwise it will be blue
        RGB_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        RGB_extracted_image = cv2.cvtColor(img_extracted_field, cv2.COLOR_BGR2RGB)
        cv2.imshow('Output', RGB_img)
        cv2.imshow('Croped', RGB_extracted_image)
        cv2.waitKey(0)
    return fields


