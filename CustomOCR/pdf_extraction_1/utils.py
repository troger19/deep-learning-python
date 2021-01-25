import re
import json
import re
import unidecode
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import cv2
from CustomOCR.pdf_extraction_1.utils import *
import requests
from os.path import splitext
import pytesseract
import numpy as np
import csv

amount_whitelist_characters = re.compile('[^0-9,.]')
reg_total_amount = 'UHRADE(.*)(e|eur|EUR|€|euro)(?:\s|$)'
reg_total_amount1 = 'UHRAD(.*)(e|eur|EUR|€|euro)(?:\s|$)\d+\s*\d+\s*\d+[.,]?\d+'
reg_ecv = '(B(A|B|C|J|L|N|R|S|Y|T)|CA|D(K|S|T)|G(A|L)|H(C|E)|IL|K(A|I|E|K|M|N|S)|L(E|C|M|V)|M(A|I|L|T|Y)|N(I|O|M|R|Z)|P(B|D|E|O|K|N|P|T|U|V)|R(A|K|S|V)|S(A|B|C|E|I|K|L|O|N|P|V)|T(A|C|N|O|R|S|T|V)|V(K|T)|Z(A|C|H|I|M|V))([ |-]{0,1})([0-9]{3})([A-Z]{2})'
reg_iban = 'SK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}|SK\d{22}'
reg_vin = '(([a-h,A-H,j-n,J-N,p-z,P-Z,0-9]{9})([a-h,A-H,j-n,J-N,p,P,r-t,R-T,v-z,V-Z,0-9])([a-h,A-H,j-n,J-N,p-z,P-Z,0-9])\s*(\d{6}))'
reg_total_amount_number = ['\d+\s*\d+\s*\d+[.,]?\d+','\d+[.,]?\d+']

def extract_pdf_text(unaccented_upper_text):
    extracted_values = {}

    # SUMA
    amount_str = re.search(reg_total_amount, unaccented_upper_text)
    if not amount_str:
        amount_str = re.search(reg_total_amount1, unaccented_upper_text)
    if amount_str:
        m = re.search('\d', amount_str.group())
        if m:
            amount_str = amount_str.group()[m.start():]
            amount = re_replace(amount_str)
            # print(amount)
            extracted_values.update({'suma': amount})

    # ECV
    license_plate_number = re.search(reg_ecv, unaccented_upper_text)
    if license_plate_number:
        # print(license_plate_number.group())
        extracted_values.update({'ecv': license_plate_number.group()})

    # IBAN
    IBAN = re.search(reg_iban, unaccented_upper_text)
    if IBAN:
        # print(IBAN.group())
        extracted_values.update({'iban': IBAN.group()})

    # VIN
    vin = re.findall(reg_vin, unaccented_upper_text)
    if vin:
        for i, v in enumerate(vin):
            print(vin[i][0])
            if (re.search('SK\d{2}', vin[i][0])):
                continue
            else:
                extracted_values.update({'vin': vin[i][0].replace(' ','')})
        # print(vin.group())


    return extracted_values


def re_replace(string):
    return re.sub(amount_whitelist_characters, '', string)


def load_target_values(filename):
    with open('validation\\validation.json', encoding='utf-8') as f:
        template = json.load(f)
        for i, file in enumerate(template['files']):
            if file['filename'] == filename:
                return file['values']
        else:
            return {}


def calculate_accuracy(filename, extracted_values,elapsed_time):
    print('extracted_values =>', extracted_values)

    target_values = load_target_values(filename)
    if not target_values:
        print('Nepodarilo sa nacitat cielove hodnoty pre porovnanie')
    else:
        print('target_values =>', target_values)
        count = 0
        for key in target_values.keys():
            if target_values.get(key) == extracted_values.get(key):
                count += 1
        accuracy = count / len(target_values.keys()) * 100
        print('presnost = ', accuracy, ' %')
    save_to_csv(filename,target_values,extracted_values,elapsed_time)


def extract_qr_code(full_path):
    extracted_values = {}
    extension = splitext(full_path)[1]
    qr = None
    if extension in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
        img = cv2.imread(full_path)
        qr = decode(img)
    elif extension == '.pdf':
        img = convert_from_path(full_path)
        for pageNumber, page in enumerate(img):
            qr = decode(page)
    else:
        print('Zadany format dokumentu nie je podporovany')

    if qr and qr[0][1]=='QRCODE':  # ak naslo zober prvy
        qrcode = qr[0].data.decode('utf-8')
        ekasa_response = requests.post('https://ekasa.financnasprava.sk/mdu/api/v1/opd/receipt/find',
                                       json={"receiptId": qrcode})
        if ekasa_response and ekasa_response.json()['receipt']:
            extracted_values.update({'suma': str(ekasa_response.json()['receipt']['totalPrice']).replace('.', ',')})

    return extracted_values


def extract_dynamic_fields(image, phrases):
    dynamic_fields = {}
     ### !!! TODO tu treba loopovat bacha, iba prva strana
    height, width = image.shape[:2]
    # converting image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)[1]
    threshold_img1 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    dst = cv2.resize(threshold_img, (int(width / 2), int(height / 2)))
    # cv2.imshow('threshold_img ',dst)
    # cv2.waitKey(0 )
    # extract fields
    # gfg = pytesseract.image_to_data(threshold_img, lang='SLK')
    extracted_dynamic_fields = pytesseract.image_to_data(threshold_img, lang='SLK', output_type='data.frame')
    extracted_dynamic_fields_temp = pytesseract.image_to_data(threshold_img, lang='SLK')
    extracted_dynamic_fields1 = pytesseract.image_to_data(threshold_img1, lang='SLK', output_type='data.frame')
    extracted_dynamic_fields1_temp = pytesseract.image_to_data(threshold_img1, lang='SLK')
    print(extracted_dynamic_fields_temp)
    # print(extracted_dynamic_fields1_temp)
    # add as key:value pair
    for i, (key,value) in enumerate(phrases.items()):
        target_word = check_and_extract(key,value, extracted_dynamic_fields)
        if not target_word:
            target_word = check_and_extract(key,value, extracted_dynamic_fields1)

        if target_word is not None:
            # dynamic_fields.update({phrase['name']: target_word})
            dynamic_fields.update({key: target_word})

    # return JSON like object with phrase name and extracted values (amounts in EUR, etc..)  -> sum_total:20,26
    return dynamic_fields

def check_and_extract(key,phrase, extracted_dynamic_fields):
    # extract the correct row and all words in a row by searching for the template phrase
    correct_row = extract_correct_row(extracted_dynamic_fields, phrase)
    # find final value in a row through regexp
    if correct_row:
        return find_final_value(str(correct_row),key)


# find the value based on defined type
def find_final_value(row, template_type):
    if template_type =='suma':
        amount = try_multiple_regex(row, reg_total_amount_number)
        return amount.replace('.',',')
    elif template_type =='iban':
        row1 = re.sub(r'[^a-zA-Z0-9]', '', row)
        return re.search(reg_iban, str(row1)).group()
    elif template_type =='vin':
        return re.search(reg_vin, str(row)).group()
    elif template_type == 'ecv':
        return re.search(reg_ecv, str(row)).group()


def extract_correct_row(extracted_dynamic_fields, phrase):
    # split tepmlate phrase into words
    phrase_split = phrase.split()
    text_split = []
    for i in phrase_split:
        text_split.append(i)
        text_split.append(i+':')
    extracted_dynamic_fields['text'] = extracted_dynamic_fields['text'].str.upper()
    for word in text_split:
        top_values = extracted_dynamic_fields[extracted_dynamic_fields['text'] == word]['top'].values
        if len(top_values) == 1:
            break
        if len(top_values) > 1:
            print('Naslo sa viac hodnot pre frazu : ' + phrase)  # TODO co robit v takomto pripade
            return None
    rows = {}
    # loop through all TOP coordinates of the 1. word
    for top in top_values:
        # create the range of TOP pixels for 1. word, because words in row can be upper or lower case and therefore the distance from the TOP is changing slightly <top-1, top +5>  because usualy the 1. word is Upper case
        pixels_from_top_range = np.arange(top - 8, top + 100, ).tolist()
        # all words that were found +- around same distance from Top of the page
        row_text = extracted_dynamic_fields[extracted_dynamic_fields['top'].isin(pixels_from_top_range)]['text'].values
        # after_text = extracted_dynamic_fields[extracted_dynamic_fields['left'].isin( np.arange(left_values,left_values+ 400, ).tolist())]['text'].values
        # clean the row by replacing non values and spaces
        clear_row_text = str(row_text).replace('nan', '').replace(' ', '')
        # if there is a word, then put it into the dictionary
        if len(row_text) != 0:
            rows.update({top: clear_row_text})
    # while 1 word can be found in multiple row on a page, we need to find the correct row by comparing the occurence of the phrase from template with the extracted row. If all words from template exists in the extracted row, then thats the correct row.
    for row in rows:
        # split each row into arrays so we can easily compares the subarrays
        # correct_row = rows[row].replace('\'', ' ').split()
        correct_row = rows[row].replace('\'', ' ')
        # checks if ALL words from template exists in row. If some word is missing then the not the correct row or the extraction failed.
        # is_all_words_found = all(item in correct_row for item in phrase['text'].split())
        # if is_all_words_found:
        if True:
            # if all words matched then its correct row
            return correct_row


def try_multiple_regex(row, regexp):
    final_extraction = None
    for i in regexp:
        final_extraction = re.search(i, str(row))
        if final_extraction:
            final_extraction = final_extraction.group().replace(' ', '')
            break

    return final_extraction

def save_to_csv(filename,target_values, extracted_values,elapsed_time):
    csv_columns= ['filename', 'duration','method','suma_target', 'suma_extracted']
    dict_data = [
        {'filename': filename,'duration':elapsed_time, 'method':'OCR','suma_target': target_values.get('suma',' -'), 'suma_extracted': extracted_values.get('suma',' -')}
    ]
    csv_file = "data.csv"
    try:
        with open(csv_file, 'a',newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

