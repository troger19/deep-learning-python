import pdfplumber
import re
import unidecode
import PyMuPDF

from CustomOCR.pdf_extraction_1.utils import *

path = '..\\..\\Datasets\\faktury\\faktury_csob\\'
filename = 'autonova.pdf'
# filename = 'impa.pdf'

percentage = get_text_percentage(path + filename)
print(percentage)

extracted_values ={}

with pdfplumber.open(path+filename) as pdf:
    first_page = pdf.pages[0]
    extracted_text = first_page.extract_text()
    if extracted_text:
        print('Extrahujem text z PDF')
        unaccented_upper_text = unidecode.unidecode(extracted_text.upper())
        extracted_values = extract_pdf_text(unaccented_upper_text)
    # print(unaccented_upper_text)
    else:
        print('pouzivam OCR')


target_values = load_target_values(filename)

print('target_values =>', target_values)
print('extracted_values =>',extracted_values)

accuracy = calculate_accuracy(target_values, extracted_values)
print('presnost = ', accuracy, ' %')
