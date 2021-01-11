import utils

pdfko = '..\\..\\Datasets\\faktury\\pdf\\orange\\1.pdf'

text = utils.convert_pdf_to_string(pdfko)
print(text)

