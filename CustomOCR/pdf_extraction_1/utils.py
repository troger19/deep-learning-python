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
import xlrd
import openpyxl

amount_whitelist_characters = re.compile('[^0-9,.]')
reg_total_amount2 = ['UHRADE(.*)(e|eur|EUR|€|euro)(?:\s|$)', 'SUMA(.*)(e|eur|EUR|€|euro)(?:\s|$)']

reg_ecv = '(B(A|B|C|J|L|N|R|S|Y|T)|CA|D(K|S|T)|G(A|L)|H(C|E)|IL|K(A|I|E|K|M|N|S)|L(E|C|M|V)|M(A|I|L|T|Y)|N(I|O|M|R|Z)|P(B|D|E|O|K|N|P|T|U|V)|R(A|K|S|V)|S(A|B|C|E|I|K|L|O|N|P|V)|T(A|C|N|O|R|S|T|V)|V(K|T)|Z(A|C|H|I|M|V))([ |-]{0,1})([0-9]{3})([A-Z]{2})'
reg_iban = '[5S]K\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}|SK\d{22}'
reg_vin = '(([a-h,A-H,j-n,J-N,p-z,P-Z,0-9]{9})([a-h,A-H,j-n,J-N,p,P,r-t,R-T,v-z,V-Z,0-9])([a-h,A-H,j-n,J-N,p-z,P-Z,0-9])\s*(\d{6}))'
reg_total_amount_number = ['\d+\s*\d+\s*\d+[.,]?\d+', '\d+[.,]?\d+']
reg_ico = ['ICO\s*?.?\s*?(\d{8})','ICO\s*?.?\s*?(\d{2} \d{3} \d{3})']


def extract_pdf_text(unaccented_upper_text, extracted_values):
    extracted_values = extracted_values
    # SUMA
    amount_str = try_multiple_regex(unaccented_upper_text, reg_total_amount2)
    if amount_str:
        m = re.search('\d', amount_str)
        if m:
            amount_str = amount_str[m.start():]
            amount = re_replace(amount_str)
            # print(amount)
            extracted_values.update({'cena_s_dph': amount})

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
                extracted_values.update({'vin': vin[i][0].replace(' ', '')})
        # print(vin.group())

    # ICO
    ICO = try_multiple_regex(unaccented_upper_text, reg_ico)
    if ICO:
        extracted_values.update({'ico': ICO})

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


def load_target_values_excel1():
    ps = openpyxl.load_workbook('validation\\Faktury-template.xlsx')
    sheet = ps['Sheet1']
    print(sheet.max_row)
    total_info = {}
    for row in range(2, sheet.max_row + 1):
        filename = sheet['A' + str(row)].value
        nazov = sheet['B' + str(row)].value
        ico = sheet['C' + str(row)].value
        iban = sheet['D' + str(row)].value
        cislo_fakt = sheet['E' + str(row)].value
        datum_dodania = sheet['F' + str(row)].value
        datum_vyhotovenia = sheet['G' + str(row)].value
        cez_bez_dph = sheet['H' + str(row)].value
        cena_s_dph = sheet['I' + str(row)].value
        mena = sheet['J' + str(row)].value
        ecv = sheet['K' + str(row)].value
        vin = sheet['L' + str(row)].value

        total_info.setdefault(filename, {'nazov': None,
                                         'ico': '',
                                         'iban': '',
                                         'cislo_fakt': '',
                                         'datum_dodania': '',
                                         'datum_vyhotovenia': '',
                                         'cez_bez_dph': '',
                                         'cena_s_dph': '',
                                         'mena': '',
                                         'ecv': '',
                                         'vin': ''
                                         })

        total_info[filename]['nazov'] = nazov
        total_info[filename]['ico'] = ico
        total_info[filename]['iban'] = iban
        total_info[filename]['cislo_fakt'] = cislo_fakt
        total_info[filename]['datum_dodania'] = datum_dodania
        total_info[filename]['datum_vyhotovenia'] = datum_vyhotovenia
        total_info[filename]['cez_bez_dph'] = cez_bez_dph
        total_info[filename]['cena_s_dph'] = None if cena_s_dph is None else str(cena_s_dph).replace('.', ',')
        total_info[filename]['mena'] = mena
        total_info[filename]['ecv'] = ecv
        total_info[filename]['vin'] = vin

    return total_info


def calculate_accuracy(filename,target_values, extraction_method, extracted_values, elapsed_time):
    print('extracted_values =>', extracted_values)
    if not target_values:
        print('Nepodarilo sa nacitat cielove hodnoty pre porovnanie')
    else:
        print('target_values =>', target_values)
        count_hit = 0
        count_all = 0
        for key in target_values.keys():
            if target_values.get(key) is not None:
                count_all += 1
                if target_values.get(key) == extracted_values.get(key):
                    count_hit += 1
        accuracy = count_hit / count_all * 100
        print('presnost = ', accuracy, ' %')
    save_to_csv(filename, extraction_method, target_values, extracted_values, elapsed_time)


def extract_qr_code(full_path,extraction_method):
    extracted_values = {}
    extension = splitext(full_path)[1]
    qr = None
    if extension in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
        img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), -1)
        qr = decode(img)
    elif extension == '.pdf':
        img = convert_from_path(full_path)
        for pageNumber, page in enumerate(img):
            qr = decode(page)
    else:
        print('Zadany format dokumentu nie je podporovany')

    if qr and qr[0][1] == 'QRCODE':  # ak naslo zober prvy
        qrcode = qr[0].data.decode('utf-8')
        if re.match('^SPD\*\d\.\d\*', qrcode):   # QR kod typu SPD - Short Payment Descriptor
            extraction_method = 'QR - SPD'
            extract_spd(qrcode,extracted_values)
        else:
            ekasa_response = requests.post('https://ekasa.financnasprava.sk/mdu/api/v1/opd/receipt/find',
                                       json={"receiptId": qrcode})
            if ekasa_response and ekasa_response.json()['receipt']:
             extracted_values.update({'cena_s_dph': str(ekasa_response.json()['receipt']['totalPrice']).replace('.', ',')})
             extraction_method = 'QR - Ekasa'
            else:
                print('PAY-By-Square')
                extraction_method = 'QR - PayBySquare'


    return extracted_values,extraction_method


# vytiahni potrebne fieldy z SPD formatu, ktory je v tvare SPD*1.0*ACC:CZ64030*AM:6897*CC:CZK*DT:2020031....
def extract_spd(qrcode,extracted_values):
    split = qrcode.split('*')
    spd = {}
    for s in split[2:-1]:
        s_split = s.split(':')
        spd.update({s_split[0]: s_split[1]})
    extracted_values.update({'cena_s_dph':spd['AM']})


# dynamicke vytahovanie hodnot podla regex
def extract_dynamic_fields(image, phrases, extracted_values):
    dynamic_fields = extracted_values
    if (len(image.shape) >2):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    height, width = image.shape[:2]
    # converting image into gray scale image

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
    # print(extracted_dynamic_fields_temp)
    # print(extracted_dynamic_fields1_temp)
    # add as key:value pair
    for i, (key, value) in enumerate(phrases.items()):
        target_word = check_and_extract(key, value, extracted_dynamic_fields)
        if not target_word:
            target_word = check_and_extract(key, value, extracted_dynamic_fields1)

        if target_word is not None:
            dynamic_fields.update({key: target_word})

    # return JSON like object with phrase name and extracted values (amounts in EUR, etc..)  -> sum_total:20,26
    return dynamic_fields


# extrahovanie textu cez Tesseract podla klucovych slov. Extrahovanie konkretneho riadku a vyparsovanie cielovej hodnoty
def check_and_extract(key, phrase, extracted_dynamic_fields):
    if not extracted_dynamic_fields['text'].isnull().values.all():
        correct_row = extract_correct_row(extracted_dynamic_fields, phrase)
        if correct_row:  # find final value in a row through regexp
            return find_final_value(str(correct_row), key)
    else:
        return '-'


# hladanie konkretnej hodnoty podla zadefinovaneho typu. Vstupom je cely riadkok extrahovaneho textu, v ktorom by sa mala nachadzat hladana hodnota
def find_final_value(row, template_type):
    if template_type == 'cena_s_dph':
        amount = try_multiple_regex(row, reg_total_amount_number)
        if amount:
            return amount.replace('.', ',')
        else:
            return None
    elif template_type == 'iban':
        row1 = re.sub(r'[^a-zA-Z0-9]', '', row)
        re_search = re.search(reg_iban, str(row1))
        if re_search:
            group = re_search.group()
            if group.startswith('5'):
                group = 'S' + group[1:]
            return group
    elif template_type == 'vin':
        search = re.search(reg_vin, str(row))
        if search:
            return search.group()
        else:
            return None
    elif template_type == 'ecv':
        search = re.search(reg_ecv, str(row))
        if search:
            return search.group()
        else:
            return None


# hlada sa dana fraza ku ktorej sa priradi jej koordinat TOP.. nasledne sa vytiahnu vsetky slova v riadku podla predpokladu, ze ich TOP hodnota pripadne do intervalu [TOP-8, TOP+100]
def extract_correct_row(extracted_dynamic_fields, phrase):
    phrase_split = phrase.split()     # split search phrase into words
    text_split = []
    for i in phrase_split:
        text_split.append(i)
        text_split.append(i + ':')
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


# metoda na vyskusanie hladania pre pole regexpov.. pozor, poradie je dolezite, ak sa slovo najde loop konci, preto treba zadavat regexy od najvaic specifickej k najmenej specifickej
def try_multiple_regex(row, regexp):
    final_extraction = None
    for i in regexp:
        final_extraction = re.search(i, str(row))
        if final_extraction:
            final_extraction = final_extraction.group(1).replace(' ', '')
            break

    return final_extraction


#ulozenie extahovanych aj cielovych hodnot do vysledneho suboru kvoli analyze presnosti extrahovania
def save_to_csv(filename, extraction_method, target_values, extracted_values, elapsed_time):
    csv_columns = ['filename', 'duration', 'method', 'ico_target', 'ico_extracted','cena_s_dph_target', 'cena_s_dph_extracted', 'iban_target', 'iban_extracted',
                   'ecv_target', 'ecv_extracted', 'vin_target', 'vin_extracted']
    dict_data = [
        {'filename': filename, 'duration': elapsed_time, 'method': extraction_method,
         'ico_target': target_values.get('ico', ' -'), 'ico_extracted': extracted_values.get('ico', ' -'),
         'cena_s_dph_target': target_values.get('cena_s_dph', ' -'), 'cena_s_dph_extracted': extracted_values.get('cena_s_dph', ' -'),
         'iban_target': target_values.get('iban', ' -'), 'iban_extracted': extracted_values.get('iban', ' -'),
         'ecv_target': target_values.get('ecv', ' -'), 'ecv_extracted': extracted_values.get('ecv', ' -'),
         'vin_target': target_values.get('vin', ' -'), 'vin_extracted': extracted_values.get('vin', ' -'),
         }]
    csv_file = "data.csv"
    try:
        with open(csv_file, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
