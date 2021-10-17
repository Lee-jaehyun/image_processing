from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import optimizers
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize
import time
import datetime


train_dir = 'data/train'
label = sorted(next(os.walk(train_dir))[1])

group = {}
for i in range(len(label)):
    group[i] = label[i]


file_list = os.listdir('../test_machine')
path = '../test_machine/'


### load model
model1 = load_model('1017_mobile_output/machine_classification_m1017_0001_224.h5')
#start = time.time()

#model1.summary()

#test_img = image.load_img('train/가리비/A270332XX_00983.jpg', target_size=(150, 150))

#test_img = image.load_img('/home/ljh/Desktop/image_processing-main/food_classification/data/train/혈압계/k001.jpg', target_size=(250, 250))
#test_img = image.load_img('data/train/혈당계/j001.jpg', target_size=(250, 300))
for i in range(len(file_list)):

#test_img = image.load_img('../test_machine/C_hulap.jpg', target_size=(224, 224))
    start = time.time()
    test_img = image.load_img(path+file_list[i], target_size=(224, 224))




#plt.imshow(test_img)
#plt.show()

#train_dir = 'data/train'
#label = sorted(next(os.walk(train_dir))[1])


    test_num = np.asarray(test_img, dtype=np.float32)
    test_num = np.expand_dims(test_num, axis=0)
    test_num = test_num / 255
#print(test_num.shape)
#print(type(test_num))

    print(model1.predict(test_num))
#print(model1.predict_classes(test_num))
    print(np.argmax(model1.predict(test_num), axis=-1))
    print("==================================")
    print("[INFO] :", group[int(np.argmax(model1.predict(test_num), axis=-1))])

    end = time.time()
    sec = (end - start)
    print("TOTAL_process :", datetime.timedelta(seconds=sec))


    plt.imshow(test_img)
    plt.show()