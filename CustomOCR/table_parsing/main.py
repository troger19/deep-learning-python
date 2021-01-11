import camelot
tables = camelot.read_pdf('camelot.pdf')
print(tables)

# tables.export('camelot.csv', f='csv', compress=True) # json, excel, html, sqlite
# print(tables[0])

# tables[0].to_csv('camelot.csv') # to_json, to_excel, to_html, to_sqlite
# print(tables[0].df) # get a pandas DataFrame!