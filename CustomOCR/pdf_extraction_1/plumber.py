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
import openpyxl

#TODO
#https://b2bportaltest.csobpoistovna.sk/test/api/skp/swagger-ui.html#/SkpRestController/getVehicleEvidenceUsingGET

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
phrases_to_extract = {'cena_s_dph': 'ÚHRADE UHRADE CELKOM SUMA ÚHRADÉ CELKEM', 'iban': 'IBAN','ico': 'ICO IČO','ecv': 'ECV EČV SPZ ŠPZ EVČ EČ','vin': 'VIN'}
myList = ['ÚHRADE','SUMA','CELKOM','FAKTURA','FAKTÚRA','IBAN']
target_values = 'validation\\faktury_1000.xlsx'


def are_all_values_extracted(extracted_values):
    if extracted_values.get('iban') is not None and extracted_values.get('ico') is not None:
        ico_from_iban = ico_servisy.get(extracted_values.get('iban'))
        ico_extracted = extracted_values.get('ico')
        if ico_extracted !=ico_from_iban and ico_from_iban is not None:
            extracted_values.update({'ico': ico_from_iban})
    if (extracted_values.get('cena_s_dph') is not None and extracted_values.get('ico') is not None and extracted_values.get('iban') is not None
            and extracted_values.get('ecv') is not None):
        if 'HRAD' in extracted_values.get('cena_s_dph_word'):
            hodnoty = extracted_values.get('cena_s_dph_word').split(';')
            for h in hodnoty:
                if 'HRAD' in h:
                    cena_uhrady = h.split(':')
                    extracted_values.update({'cena_s_dph':cena_uhrady[2] })
                    break
            return True
        else:
            return False
    else:
        return False

def extract_values_from_file(full_path):
    img_pdf = None
    start_time = time.time()
    extraction_method =None
# pokladnicny blok
    extracted_values,extraction_method = extract_qr_code(full_path,extraction_method)

    extension = splitext(full_path)[1]
    if not are_all_values_extracted(extracted_values):
        if extension.lower() =='.pdf':
        # faktura
            try:
                with pdfplumber.open(full_path) as pdf:
                    for i,page in enumerate(pdf.pages[:2]):
                        first_page =page
                        extracted_text = first_page.extract_text()
                        if bool(extracted_text) and any(char.isdigit() for char in extracted_text) and len(re.findall('(cid:\d+?)',extracted_text)) <100 and any(x in extracted_text.upper() for x in myList):
                        # if False:
                            print('Extrahujem text z PDF')
                            extraction_method = set_extraction_method(extracted_values, extraction_method, 'PDF TEXT')
                            unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
                            # print(unaccented_upper_text)
                            extracted_values = extract_pdf_text(unaccented_upper_text,extracted_values,ico_servisy)
                            if not are_all_values_extracted(extracted_values):
                                extracted_values, extraction_method = ocr_extraction(extracted_values, extraction_method, full_path, i, img_pdf)
                        else:
                            extracted_values, extraction_method = ocr_extraction(extracted_values, extraction_method, full_path, i, img_pdf)
                        if are_all_values_extracted(extracted_values):
                            break
            except:
                print('Nastal problem pri spracovani PDF' + full_path)
                extraction_method = 'ERROR'
        elif extension.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
            extraction_method = set_extraction_method(extracted_values, extraction_method,'OCR')
            img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), -1)
            extracted_values = extract_dynamic_fields(img, phrases_to_extract,extracted_values,ico_servisy,os.path.basename(full_path))
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


def ocr_extraction(extracted_values, extraction_method,full_path, i, img_pdf):
    print('pouzivam OCR')
    extraction_method = set_extraction_method(extracted_values, extraction_method, 'OCR')
    if img_pdf is None:
        img_pdf = convert_from_path(full_path)
    extracted_values = extract_dynamic_fields(np.array(img_pdf[i]), phrases_to_extract, extracted_values, ico_servisy,
                                              os.path.basename(full_path))
    return extracted_values,extraction_method


def set_extraction_method(extracted_values, extraction_method, new_method):
    if len(extracted_values) > 0:  # ak uz bolo nieco rozpoznane inou metodou zachovaj aj povodnu metodu
        extraction_method = extraction_method + ' | ' + new_method
    else:  # ak zatial nebolo nic rozpoznane, urci novu metodu
        extraction_method = new_method
    return extraction_method


# zaciatok
start_time = time.time()
all_target_values = load_target_values_excel(target_values)
ico_servisy = load_ico_servisy()
for i,y in enumerate(invoices_list):
    print(y)
    extract_values_from_file(path+y)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)
#koniec
