import re
import json

amount_whitelist_characters = re.compile('[^0-9,.]')
reg_total_amount = 'UHRADE(.*)(e|eur|EUR|€|euro)(?:\s|$)'
reg_ecv = '(B(A|B|C|J|L|N|R|S|Y|T)|CA|D(K|S|T)|G(A|L)|H(C|E)|IL|K(A|I|E|K|M|N|S)|L(E|C|M|V)|M(A|I|L|T|Y)|N(I|O|M|R|Z)|P(B|D|E|O|K|N|P|T|U|V)|R(A|K|S|V)|S(A|B|C|E|I|K|L|O|N|P|V)|T(A|C|N|O|R|S|T|V)|V(K|T)|Z(A|C|H|I|M|V))([ |-]{0,1})([0-9]{3})([A-Z]{2})'
reg_iban = 'SK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}|SK\d{22}'
reg_vin = '^([A-HJ-NPR-Z\d]{3})([A-HJ-NPR-Z\d]{5})([\dX])(([A-HJ-NPR-Z\d])([A-HJ-NPR-Z\d])([A-HJ-NPR-Z\d]{6}))$'

def extract_pdf_text(unaccented_upper_text):
    extracted_values = {}

    # SUMA
    amount_str = re.search(reg_total_amount, unaccented_upper_text)
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

    #VIN
    vin = re.search(reg_vin, unaccented_upper_text)
    if vin:
        # print(vin.group())
        extracted_values.update({'vin': vin.group()})

    return extracted_values

def re_replace(string):
    return re.sub(amount_whitelist_characters, '', string)

def load_target_values(filename):
    with open('validation\\validation.json', encoding='utf-8') as f:
        template = json.load(f)
        for i, file in enumerate(template['files']):
            if file['filename'] == filename:
                return file['values']


def calculate_accuracy(target, actual):
    count = 0
    for key in target.keys():
        if target.get(key) == actual.get(key):
            count +=1
    return count/len(target.keys())*100


import fitz



def get_text_percentage(file_name: str) -> float:
    """
    Calculate the percentage of document that is covered by (searchable) text.

    If the returned percentage of text is very low, the document is
    most likely a scanned PDF
    """
    total_page_area = 0.0
    total_text_area = 0.0

    doc = fitz.open(file_name)

    for page_num, page in enumerate(doc):
        total_page_area = total_page_area + abs(page.rect)
        text_area = 0.0
        for b in page.getTextBlocks():
            r = fitz.Rect(b[:4])  # rectangle where block text appears
            text_area = text_area + abs(r)
        total_text_area = total_text_area + text_area
    doc.close()
    return total_text_area / total_page_area
print("not fully scanned PDF - text is present")