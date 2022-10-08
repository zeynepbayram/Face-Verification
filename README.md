# Face-Verification
![image](https://user-images.githubusercontent.com/86806643/194709809-2561f686-9f04-4ac1-b569-6d26fbbb5280.png)
![image](https://user-images.githubusercontent.com/86806643/194709978-db7795fc-6143-4928-a511-6572959094d3.png)
![image](https://user-images.githubusercontent.com/86806643/194709994-527228ef-5e02-4c33-9524-222d0768551a.png)
![image](https://user-images.githubusercontent.com/86806643/194710029-6ec67914-5918-4414-84f0-14077ca60891.png)
![image](https://user-images.githubusercontent.com/86806643/194710141-b99d004d-89d5-4e8e-9c3e-bd1878d531b0.png)
```python
   def __init__(self, epsilon = 0.40): 
        print("Building Caffe Face Detector..") 
        self.face_detector = cv2.dnn.readNetFromCaffe("C:/Users/Zeynep/Desktop/SVMfaceR/deploy.prototxt.txt", "C:/Users/Zeynep/Desktop/SVMfaceR/res10_300x300_ssd_iter_140000.caffemodel") 
        print("Building Verifier..") 
        self.verifier = load_model("busonmodel.h5") 
        self.epsilon = epsilon
```
* Caffe Face Detector modeli, ve daha önce kaydedilen vgg-face verification modeli yükleniyor.

```python 
def preprocess_image_rt(self, image): 
        img = cv2.resize(image, (224, 224)) 
        img = img_to_array(img) 
        img = np.expand_dims(img, axis=0) 
        img = preprocess_input(img) 
        return img
```
* VGG modeli 224x224x3 boyutlu girdileri kabul ettiği için modele verilecek görüntüler resize ediliyor.
* PIL görseli numpy dizisine çevriliyor. (img_to_array)
* preprocess_input işlemiyle görüntü normalize ediliyor.

```python 
 def findCosineSimilarity(self, source_representation, test_representation): 
        a = np.matmul(np.transpose(source_representation), test_representation) 
        b = np.sum(np.multiply(source_representation, source_representation)) 
        c = np.sum(np.multiply(test_representation, test_representation)) 
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
```

* ![image](https://user-images.githubusercontent.com/86806643/194710474-193c4185-75a8-4a54-8988-dff60502e2e9.png)
* Cosine Similarity Formulünü kullanarak verilen 2 girdinin benzerliği hesaplanıyor.
* 1'den cosine similarity değerini (benzerlik oranı) çıkardığımızda ise girdilerin farkı bulunuyor.(Cosine Distance)

```python 
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
```
* Face verification aşamasına geçebilmek için daha önce yüklenilen caffe modeli kullanılarak face detect edilir.

```python 
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
```

* Model çıktısının uzaklıkları belirlenen epsilon değerinden küçükse 1 döndür ve verification işlemi tamamlanır. Değilse 0 döndürür.

```python 
    def barcode(self, frame): 
            for code in decode(frame): 
                return code.data.decode("utf-8")
```
* Pyzbar library fonksiyonları kullanılarak barkod decode edilir.

***Test Aşaması***
Modeli test etmek için küçük ve etiketsiz bir dataset kullanıldı. Görseller aynı veya farklı insanlara ait olması durumlarına karşılık 1 - 0 şeklinde etiketlendi. Her bir görsel (kendisi dahil) birbiriyle karşılaştırıldı.

Dataset:
![image](https://user-images.githubusercontent.com/86806643/194710617-85441116-d7e7-4917-872e-2165b6f0b1a8.png)
```python 
from vggfacecomparison import FaceVerification
import os
import cv2 as cv
import numpy as np
fv = FaceVerification(0.35)
dataset_path = "C:/Users/Zeynep/Desktop/dataset"
img_names = os.listdir(dataset_path)
labels = []
distances = []
predictions= []
truths = []
names = []
count = 0
for i in range(len(img_names)):
    for j in range(len(img_names)):
        im1 = img_names[i]
        im2 = img_names[j]
        name = im1 + '-' +im2
        names.append(name)
        count +=1
        abs_im1 = os.path.join(dataset_path, im1)
        abs_im2 = os.path.join(dataset_path, im2)
        readim1 = cv.imread(abs_im1)
        readim2 = cv.imread(abs_im2)
        prediction, distance = fv.verifyFace(readim1, readim2)
        predictions.append(prediction)
        distances.append(distance)
        label = 0
        if im1[0] == im2[0]:
            label = 1
        truth = 1 if label == prediction else 0
        labels.append(label)
        truths.append(truth)
        if count %100 ==0:
            print(count)
distances = np.array(distances)
labels = np.array(labels)
truths = np.array(truths)
names = np.array(names)
predictions = np.array(predictions)
np.save('predictions.npy', predictions)
np.save('distances.npy', distances)
np.save('labels.npy', labels)
np.save('truths.npy', truths)
np.save('names.npy', names)
```

"Names" sütununda karşılaştırılan görsel isimlerine
"Labels" sütununda atanan etiketlere
"Predictions" sütununda modelin çıktısına
"Distances " sütununda predictionda kullanılan cosine distance değerine
"Truths" sütununda XNOR mantığıyla etiketlemelerin ve modelin çıktısının uyumuna yer verilmiştir.

![image](https://user-images.githubusercontent.com/86806643/194710675-1a82c308-df53-4c3f-8bc7-e29198e51450.png)

* Gerçekleştirilen 946 karşılaştırmada 63 yanlış tahmin yapıldı.
* %93,33 oranında doğruluğa sahip olduğu görüldü.
* Datasetteki kişilerin genç, yaşlı, gözlüklü, gözlüksüz, sakallı, sakalsız görsellerinin karşılaştırıldığı da göz önüne alınarak değerlendirildiğinde yeterli bir orana sahiptir.
