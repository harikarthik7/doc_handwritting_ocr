import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PySide6 import *
from PySide6.QtGui import * 
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import sys
import numpy as np
import cv2
from src.ocr.normalization import word_normalization
from src.ocr import page, words
from src.ocr.tfhelpers import Model
from src.ocr.datahelpers import idx2char
from difflib import SequenceMatcher


# global select_flag
# List of medicine names
file = open("data/medicine_name.txt","r",encoding='utf-8')
medicine_names=[]
for x in file:
    medicine_names.append(x)
    


#another algo
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def select_roi(img):
    r= cv2.selectROI("Select the Medecine Name", img)
    cv2.destroyWindow("Select the Medecine Name")
    image = img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
    return image
        

def recognise(img):
    MODEL_LOC_CTC = 'models/word-clas/CTC/Classifier1'
    CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')
    img = word_normalization(
        img,
        64,
        border=False,
        tilt=False,
        hyst_norm=False)
    length = img.shape[1]
    input_imgs = np.zeros(
            (1, 64, length, 1), dtype=np.uint8)
    input_imgs[0][:, :length, 0] = img

    pred = CTC_MODEL.eval_feed({
        'inputs:0': input_imgs,
        'inputs_length:0': [length],
        'keep_prob:0': 1})[0]

    word = ''
    for i in pred:
        word += idx2char(i + 1)
    return word

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Text Recognition')
        self.setGeometry(500, 200, 900, 500)
        self.setStyleSheet("background-color: #edebca;")
        #x,y,w,h

        self.image_label = QLabel(self)
        self.image_label.setText("Upload a Handwritten Medicine Name")
        self.image_label.setStyleSheet("font-size: 20px; font-weight: bold; font-family: Helvetica; color:black")
        self.image_label.setGeometry(35, 25, 400, 25)
        
        self.upload_canvas = QFrame(self)
        self.upload_canvas.setGeometry(70, 85, 300, 300)
        self.upload_canvas.setStyleSheet("background-color: lightgray; border-radius:10px")

        self.upload_btn = (QPushButton(self))
        self.upload_btn.setText("UPLOAD")
        self.upload_btn.setGeometry(150, 410, 150, 50)
        self.upload_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px; border-radius: 10px;")
        self.upload_btn.clicked.connect(self.uploadImage)
        
        self.select_btn = (QPushButton(self))
        self.select_btn.setText("SELECT")
        self.select_btn.setGeometry(500, 45, 150, 50)
        self.select_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 14px; border-radius: 10px;")
        self.select_btn.clicked.connect(self.selectImage)
        
        self.predict_btn = (QPushButton(self))
        self.predict_btn.setText("PREDICT")
        self.predict_btn.setGeometry(700, 45, 150, 50)
        self.predict_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px; border-radius: 10px;")
        self.predict_btn.clicked.connect(self.processImage)
        
        #Processing..
        self.process_label = QLabel(self)
        self.process_label.setText("Upload the image")
        self.process_label.setStyleSheet("font-size: 30px; font-weight: bold; font-family: Helvetica; color:black")
        self.process_label.setGeometry(560, 120, 300, 50)
        
        #textbox label
        self.txt_label = QLabel(self)
        self.txt_label.setText("Predicted Medicine/s :")
        self.txt_label.setStyleSheet("font-size: 15px; font-weight: bold; font-family: Helvetica; color:black; background-color: #edebca;")
        self.txt_label.setGeometry(550, 190, 200, 25)
        
        #textbox
        self.predict_canvas = QFrame(self)
        self.predict_canvas.setGeometry(545, 230, 240, 240)
        self.predict_canvas.setStyleSheet("background-color: white; border-radius:5px")
        
        self.image_label = QLabel(self)
        self.image_label.setText("")
        self.image_label.setStyleSheet("font-size: 12px; font-weight: bold; font-family: Helvetica; color:black; background-color: white")
        self.image_label.setGeometry(555, 240, 220, 200)
        
        self.img2 = QImage("Images/black.png")
        self.img2 = self.img2.scaled(300, 300)
        self.bg2 = QPixmap.fromImage(self.img2)
        self.label2 = QLabel(self)
        self.label2.setPixmap(self.bg2)
        self.label2.setGeometry(70, 85, 300, 300)
        


    def uploadImage(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Images (*.png *.jpg *.jpeg)')
        if self.file_path:
            self.imm = QImage(self.file_path)
            self.imm = self.imm.scaled(300, 300)
            transform = QTransform().rotate(0)
            rotated_image = self.imm.transformed(transform)
            self.bg3 = QPixmap.fromImage(rotated_image)
            self.label2.setPixmap(self.bg3)
            self.img = cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)
            self.img = cv2.resize(self.img,(300,300))
        self.process_label.setText(f'Select or Predict')
        self.process_label.setGeometry(520, 120, 350, 50)
        


    def selectImage(self):
        imm = cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)
        imm=cv2.resize(imm,(1000,1000))
        # cv2.imshow("app",self.)
        self.img=select_roi(imm)
        self.image_label.setText(f' ')
            

    def processImage(self):
        self.image_label.setText(f' ')
        self.process_label.setText(f'Processing')
        self.process_label.setGeometry(560, 120, 300, 50)
        QApplication.processEvents()
        l=[]
        print(self.file_path)
        

        crop = page.detection(self.img)
        # implt(crop)
        boxes = words.detection(crop)
        lines = words.sort_words(boxes)

        for line in lines:
            self.process_label.setText(f'Processing.')
            self.process_label.setGeometry(560, 120, 300, 50)
            QApplication.processEvents()
            # print("for running")
            ocr_output = " ".join([recognise(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line])
            # print(ocr_output)
            out1=max(medicine_names, key=lambda x: similar(ocr_output.lower(), x.lower()))
            # print(max(medicine_names, key=lambda x: similar(ocr_output.lower(), x.lower())))

            l.append(out1)
            
        self.process_label.setText(f'Processed')
        strr="\n".join(l)
        print(strr)
        self.img=cv2.cvtColor(cv2.imread(self.file_path), cv2.COLOR_BGR2RGB)
        self.image_label.setText(f'{strr}')
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec())
