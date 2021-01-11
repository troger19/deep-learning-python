# d:\NN\Miniconda3\Scripts\activate.bat d:\NN\Miniconda3\ & python.exe d:\Java\labelImg\labelImg.py
# d:\NN\Miniconda3\Scripts\activate.bat d:\NN\Miniconda3\ & python.exe D:\Java\deep-learning-python\ObjectDetection\licence_plate\yolo4\test.py


import sys

print(sys.executable)
from PyQt5.QtWidgets import QApplication, QMainWindow

def window():
    app=QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200,200,300,300)
    win.setWindowTitle('Tech with janko')

    win.show()
    sys.exit(app.exec_())

window()