import utils

# pdfko = '..\\..\\Datasets\\faktury\\pdf\\orange\\1.pdf'
pdfko = '..\\..\\Datasets\\faktury\\faktury_csob\\impa.pdf'

text = utils.convert_pdf_to_string(pdfko)
print(text)
