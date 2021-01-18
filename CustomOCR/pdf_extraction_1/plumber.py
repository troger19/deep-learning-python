import pdfplumber
import re
import unidecode

pdfko = '..\\..\\Datasets\\faktury\\faktury_csob\\impa.pdf'
# pdfko = '..\\..\\Datasets\\faktury\\faktury_csob\\melisko.pdf'

amount_whitelist_characters = re.compile('[^0-9,.]')
reg_total_amount = 'UHRADE(.*)(e|eur|EUR|€|euro)(?:\s|$)'
reg_ecv = '(B(A|B|C|J|L|N|R|S|Y|T)|CA|D(K|S|T)|G(A|L)|H(C|E)|IL|K(A|I|E|K|M|N|S)|L(E|C|M|V)|M(A|I|L|T|Y)|N(I|O|M|R|Z)|P(B|D|E|O|K|N|P|T|U|V)|R(A|K|S|V)|S(A|B|C|E|I|K|L|O|N|P|V)|T(A|C|N|O|R|S|T|V)|V(K|T)|Z(A|C|H|I|M|V))([ |-]{0,1})([0-9]{3})([A-Z]{2})(?:\s|$)'
reg_iban = 'SK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}|SK\d{22}'
reg_vin = '^([A-HJ-NPR-Z\d]{3})([A-HJ-NPR-Z\d]{5})([\dX])(([A-HJ-NPR-Z\d])([A-HJ-NPR-Z\d])([A-HJ-NPR-Z\d]{6}))$'

def re_replace(string):
    return re.sub(amount_whitelist_characters, '', string)

with pdfplumber.open(pdfko) as pdf:
    first_page = pdf.pages[0]
    extracted_text = first_page.extract_text()
    unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
    print(unaccented_upper_text)

amount_str = re.search(reg_total_amount, unaccented_upper_text)
if amount_str:
    m = re.search('\d', amount_str.group())
    if m:
        amount_str = amount_str.group()[m.start():]
        amount = re_replace(amount_str)
        print(amount)
    # print(amount_str)


license_plate_number= re.search(reg_ecv, unaccented_upper_text)
if license_plate_number:
    print(license_plate_number.group())

IBAN = re.search(reg_iban, unaccented_upper_text)
if IBAN:
    print(IBAN.group())

vin = re.search(reg_vin, unaccented_upper_text)
if vin:
    print(vin.group())




