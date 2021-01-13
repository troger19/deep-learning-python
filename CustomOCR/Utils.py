import json
import os
import cv2
import pytesseract
import numpy as np
import re
import math


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


def extract_dynamic_fields(page, phrases):
    dynamic_fields = {}
    image = np.array(page)
    height, width = image.shape[:2]
    # converting image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)[1]
    threshold_img1 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    dst = cv2.resize(threshold_img, (int(width / 2), int(height / 2)))
    # cv2.imshow('threshold_img ',dst)
    # cv2.waitKey(0 )
    # extract fields
    gfg = pytesseract.image_to_data(threshold_img, lang='SLK')
    extracted_dynamic_fields = pytesseract.image_to_data(threshold_img, lang='SLK', output_type='data.frame')
    extracted_dynamic_fields_temp = pytesseract.image_to_data(threshold_img, lang='SLK')
    extracted_dynamic_fields1 = pytesseract.image_to_data(threshold_img1, lang='SLK', output_type='data.frame')
    # extracted_dynamic_fields1_temp = pytesseract.image_to_data(threshold_img1, lang='SLK')
    # print(extracted_dynamic_fields_temp)
    # print(extracted_dynamic_fields1_temp)
    # add as key:value pair
    for i, phrase in enumerate(phrases):
        target_word = check_and_extract(phrase, extracted_dynamic_fields)
        if not target_word:
            target_word = check_and_extract(phrase, extracted_dynamic_fields1)

        if target_word is not None:
            dynamic_fields.update({phrase['name']: target_word})

    # return JSON like object with phrase name and extracted values (amounts in EUR, etc..)  -> sum_total:20,26
    return dynamic_fields


def check_and_extract(phrase, extracted_dynamic_fields):
    # extract the correct row and all words in a row by searching for the template phrase
    correct_row = extract_correct_row(extracted_dynamic_fields, phrase)
    # find final value in a row through regexp
    final_value = find_final_value(str(correct_row),phrase)

    return final_value


def check_type(value, type):
    if type in 'decimal':
        r = re.compile(r"^\d*[.,]?\d*$")
        if r.match(value):
            return True
        else:
            print('The extracted value' + value + ' is not decimal')
            return False
    elif type in 'integer':
        if value.is_integer():
            return True
        else:
            print('The extracted value' + value + ' is not integer')
            return False

# find the value based on defined type
def find_final_value(row, template_type):
    if template_type['type'] in 'decimal':
        return re.findall('\d+[.,]\d+', str(row))
    elif template_type['type'] in 'integer':
        return re.findall('\d+', str(row))



def extract_correct_row(extracted_dynamic_fields, phrase):
    # split tepmlate phrase into words
    text_split = phrase['text'].split()
    # take first word from phrase and find its coordinates.. the word can occurs multiple time on page
    top_values = extracted_dynamic_fields[extracted_dynamic_fields['text'] == text_split[0]]['top'].values
    rows = {}
    # loop through all TOP coordinates of the 1. word
    for top in top_values:
        # create the range of TOP pixels for 1. word, because words in row can be upper or lower case and therefore the distance from the TOP is changing slightly <top-1, top +5>  because usualy the 1. word is Upper case
        pixels_from_top_range = np.arange(top - 2, top + 7, ).tolist()
        # all words that were found +- around same distance from Top of the page
        row_text = extracted_dynamic_fields[extracted_dynamic_fields['top'].isin(pixels_from_top_range)]['text'].values
        # clean the row by replacing non values and spaces
        clear_row_text = str(row_text).replace('nan', '').replace(' ', '')
        # if there is a word, then put it into the dictionary
        if len(row_text) != 0:
            rows.update({top: clear_row_text})
    # while 1 word can be found in multiple row on a page, we need to find the correct row by comparing the occurence of the phrase from template with the extracted row. If all words from template exists in the extracted row, then thats the correct row.
    for row in rows:
        # split each row into arrays so we can easily compares the subarrays
        correct_row = rows[row].replace('\'', ' ').split()
        # checks if ALL words from template exists in row. If some word is missing then the not the correct row or the extraction failed.
        is_all_words_found = all(item in correct_row for item in phrase['text'].split())
        if is_all_words_found:
            # if all words matched then its correct row
            return correct_row
