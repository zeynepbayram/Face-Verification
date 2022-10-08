from pyexpat import model
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from FaceVerification import FaceVerification


predictions = []
cap = cv2.VideoCapture(0)
ref_path = 'C:/Users/Zeynep/Desktop/images'
fv = FaceVerification()
barcode_succes = 0
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 50)
fontScale = 1
color = (60,20,220)
thickness = 2
verified = 0
while True:
    ret, frame = cap.read(0)
    if barcode_succes == 0:
        m = fv.barcode(frame)
    try:
        ref = cv2.imread(ref_path + "/" + m +'.jpg')
        barcode_succes = 1
    except:
        frame = cv2.putText(frame, 'Lutfen barkod okutunuz.', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    if barcode_succes == 1:
        try:
            predictions.append((fv.verifyFace(frame, ref))[0])
            if not verified:
                text = 'Verifyng...' 
                frame = cv2.putText(frame, text , org, font, fontScale, color, thickness, cv2.LINE_AA)
            if len(predictions) == 4:
                text = "%" + str(sum(predictions) * 25) + " verified"
                if sum(predictions) >= 4:
                    org = (30, 100)
                    frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.imshow('f', frame)
                    cv2.waitKey(5000)
                    verified = 1
                else:
                    org = (30, 100)
                    frame = cv2.putText(frame, "Verification Gerceklestirilemedi.", org, font, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.imshow('f', frame)
                    cv2.waitKey(3000)
                    verified = 1
                predictions = []
        except:
            frame = cv2.putText(frame, "Yuz Algilanamadi.", org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('f', frame)
    key = cv2.waitKey(10)
    if key == 27 or verified == 1:
        break
cap.release()