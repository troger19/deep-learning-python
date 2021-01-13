import lzma
# data = b"Welcome to TutorialsPoint"
# f = lzma.open("test.xz","wb")
# f.write(data)
# f.close()
test_str = "Ahoj Jano"
# b = bytes(test_str, 'utf-8')
b = test_str.encode('utf-8')
# res = ''.join(format(ord(i), 'b') for i in test_str)
# print(res)
fileContent = None
with open("people.xz", mode='rb') as file: # b is important -> binary
    fileContent = file.read()
decompress = lzma.decompress(fileContent)
print(decompress)
