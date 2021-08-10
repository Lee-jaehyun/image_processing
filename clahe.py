import cv2
import numpy as np
from matplotlib import pyplot as plt


## 히스토그램 균일화 ==>
# 출처 : https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html


img = cv2.imread('example999.jpeg', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img2 = clahe.apply(img)

img = cv2.resize(img, (400, 400))
img2 = cv2.resize(img2, (400, 400))

dst = np.hstack((img, img2))
cv2.imshow('img', dst)
cv2.waitKey(0)