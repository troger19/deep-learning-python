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
phrases_to_extract = {'cena_s_dph': 'SUMA ÚHRADE', 'ico': 'ICO IČO','iban': 'IBAN','vin': 'VIN','ecv': 'ECV'}


def extract_values_from_file(full_path):
    img_pdf = None
    start_time = time.time()
    extraction_method =None
# pokladnicny blok
    extracted_values,extraction_method = extract_qr_code(full_path,extraction_method)

    extension = splitext(full_path)[1]
    if not extracted_values:
        if extension =='.pdf':
        # faktura
            with pdfplumber.open(full_path) as pdf:
                for i,page in enumerate(pdf.pages):
                    first_page =page
                    extracted_text = first_page.extract_text()
                    if bool(extracted_text) and any(char.isdigit() for char in extracted_text):
                        print('Extrahujem text z PDF')
                        extraction_method = 'PDF TEXT'
                        unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
                        # print(unaccented_upper_text)
                        extracted_values = extract_pdf_text(unaccented_upper_text,extracted_values)
                    else:
                        print('pouzivam OCR')
                        extraction_method = 'OCR'
                        if img_pdf is None:
                            img_pdf = convert_from_path(full_path)
                        extracted_values = extract_dynamic_fields(np.array(img_pdf[i]), phrases_to_extract,extracted_values)
        elif extension in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
            extraction_method = 'OCR'
            img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), -1)
            extracted_values = extract_dynamic_fields(img, phrases_to_extract,extracted_values)
        else:
            print('Neznamy format vstupuneho suboru')
    # rukou pisana faktura
    elif not extracted_values:
        print('rukou pisana faktura')

    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(elapsed_time)

    target_values = all_target_values.get(os.path.basename(full_path),{})
    calculate_accuracy(os.path.basename(full_path),target_values,extraction_method, extracted_values,elapsed_time)

all_target_values = load_target_values_excel1()
for i,y in enumerate(invoices_list):
    print(y)
    extract_values_from_file(path+y)



