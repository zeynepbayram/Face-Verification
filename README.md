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
