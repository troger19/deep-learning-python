import pdfplumber
import re
import unidecode
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import cv2
from CustomOCR.pdf_extraction_1.utils import *
import requests
from os.path import splitext

path = '..\\..\\Datasets\\faktury\\faktury_csob\\'
# filename = 'peugeot.pdf'
# filename = 'melisko.pdf'
filename = 'autonova.pdf'
# filename = 'impa.pdf'
# filename = 'blocek1.jpg'
# filename = 'lidl.pdf'
full_path = path + filename

# pokladnicny blok
extracted_values = extract_qr_code(full_path)

if not extracted_values:
    # faktura
    with pdfplumber.open(full_path) as pdf:
        first_page = pdf.pages[0]
        extracted_text = first_page.extract_text()
        if extracted_text:
            print('Extrahujem text z PDF')
            unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
            extracted_values = extract_pdf_text(unaccented_upper_text)
        else:
            print('pouzivam OCR')
            pdf = convert_from_path(full_path)
            extracted_values = extract_dynamic_fields(pdf, ['úhrade','IBAN'])  # spravit dicsitoonary.. uhrada:decimal  IBAN:iban

# rukou pisana faktura
elif not extracted_values:
    print('rukou pisana faktura')


calculate_accuracy(filename, extracted_values)
