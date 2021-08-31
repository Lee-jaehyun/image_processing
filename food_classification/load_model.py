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

### load model
model1 = load_model('food_classification_m.h5')
model1.summary()


#test_img = image.load_img('train/가리비/A270332XX_00983.jpg', target_size=(150, 150))

test_img = image.load_img('../galbi2.jpeg', target_size=(150, 150))
#test_img = plt.imread('train/갈비탕/B050302_10002.jpg')
plt.imshow(test_img)
plt.show()

print(test_img)
test_num = np.asarray(test_img, dtype=np.float32)
test_num = np.expand_dims(test_num, axis=0)
test_num = test_num / 255
print(test_num.shape)
print(type(test_num))

print(model1.predict(test_num))
#print(model1.predict_classes(test_num))
print(np.argmax(model1.predict(test_num), axis=-1))