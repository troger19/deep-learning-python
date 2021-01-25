import pdfplumber
import re
import unidecode
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import cv2
from CustomOCR.pdf_extraction_1.utils import *
import requests
from os.path import splitext
import os
import time

# path = '..\\..\\Datasets\\faktury\\faktury_csob\\'
path = '..\\..\\Datasets\\faktury\\pokus\\'
# filename = 'anders.jpg'   # square
# filename = 'autofun.pdf'
# filename = 'autosklo.jpg'
# filename = 'peugeot.pdf'
# filename = 'suzuki1.pdf'
# filename = 'suzuki2.pdf'
# filename = 'suzuki3.pdf'
# filename = 'melisko.pdf'
# filename = 'autonova.pdf'
# filename = 'impa.pdf'      # 75 %
# filename = 'blocek1.jpg'
# filename = 'lidl.pdf'
# full_path = path + filename

# phrases_to_extract = {'suma': 'ÚHRADE', 'iban': 'IBAN'}
invoices_list = os.listdir(path)
phrases_to_extract = {'suma': 'SUMA ÚHRADE', 'iban': 'IBAN','vin': 'VIN','ecv': 'ECV'}

def extract_values_from_file(full_path):
    start_time = time.time()

# pokladnicny blok
    extracted_values = extract_qr_code(full_path)

    extension = splitext(full_path)[1]
    if not extracted_values:
        if extension =='.pdf':
        # faktura
            with pdfplumber.open(full_path) as pdf:
                first_page = pdf.pages[0]
                extracted_text = first_page.extract_text()
                if bool(extracted_text) and any(char.isdigit() for char in extracted_text):
                    print('Extrahujem text z PDF')
                    unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
                    print(unaccented_upper_text)
                    extracted_values = extract_pdf_text(unaccented_upper_text)
                else:
                    print('pouzivam OCR')
                    pdf = convert_from_path(full_path)
                    extracted_values = extract_dynamic_fields(np.array(pdf[len(pdf)-1]), phrases_to_extract)  # spravit dicsitoonary.. uhrada:decimal  IBAN:iban
        elif extension in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
            img = cv2.imread(full_path)
            extracted_values = extract_dynamic_fields(img, phrases_to_extract)
        else:
            print('Neznamy format vstupuneho suboru')
    # rukou pisana faktura
    elif not extracted_values:
        print('rukou pisana faktura')

    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(elapsed_time)
    calculate_accuracy(os.path.basename(full_path), extracted_values,elapsed_time)

for i,y in enumerate(invoices_list):
    extract_values_from_file(path+y)


