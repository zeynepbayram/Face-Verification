from pyexpat import model
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import os
import warnings
warnings.filterwarnings('ignore')
import numba
from numba import jit, cuda


class FaceVerification:
    @jit
    def __init__(self, epsilon = 0.40):
        print("Building Caffe Face Detector..")
        self.face_detector = cv2.dnn.readNetFromCaffe("C:/Users/Zeynep/Desktop/SVMfaceR/deploy.prototxt.txt", "C:/Users/Zeynep/Desktop/SVMfaceR/res10_300x300_ssd_iter_140000.caffemodel")

        print("Building Verifier..")
        self.verifier = load_model("busonmodel.h5")
        print(self.verifier.summary())

        self.epsilon = epsilon

    def preprocess_image_rt(self, image):
        img = cv2.resize(image, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def get_face_ssdnet(self, frame, image_size = 224):
        (h, w) = frame.shape[:2]
        resized_image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int32")
        face = frame[startY:endY, startX:endX]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (42, 64, 127), 2)
        return face

    def verifyFace(self, img1, img2):
        img1 = self.get_face_ssdnet(img1)
        img2 = self.get_face_ssdnet(img2)
        img1_representation = self.verifier.predict(self.preprocess_image_rt(img1))[0,:]
        img2_representation = self.verifier.predict(self.preprocess_image_rt(img2))[0,:]
        cosine_similarity = self.findCosineSimilarity(img1_representation, img2_representation)
        
        if(cosine_similarity < self.epsilon):
            return 1, cosine_similarity
        else:
            return 0, cosine_similarity
    
    def barcode(self, frame):
            for code in decode(frame):
                return code.data.decode("utf-8")

        