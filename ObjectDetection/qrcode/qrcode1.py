import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path

# pdf = convert_from_path('o2\\3.pdf')
pdf = convert_from_path('orange\\1.pdf')
# pdf = convert_from_path('zlavomat\\zlavomat1.pdf')

page1 = decode(pdf[0])
print(page1[0].data.decode('utf-8'))
#
opencvImage = cv2.cvtColor(np.array(pdf[0]), cv2.COLOR_RGB2BGR)
pts = np.array([page1[0].polygon], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(opencvImage, [pts], True, (255, 0, 255), 5)


#fit to the window
height, width = opencvImage.shape[:2]
dst = cv2.resize(opencvImage, (int(width/2), int(height/2)), interpolation = cv2.INTER_CUBIC)

cv2.imshow('jpg', dst)
cv2.waitKey(0)