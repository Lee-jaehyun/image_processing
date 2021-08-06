import pytesseract
import cv2
import os
from PIL import Image

image = cv2.imread("xxx.jpeg")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)

ret, img_thresh = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

'''img_thresh = cv2.adaptiveThreshold(img_blurred, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)
'''
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, img_thresh)

text = pytesseract.image_to_string(Image.open(filename), lang='kor+eng')
os.remove(filename)

print(text)
cv2.imshow('image', img_thresh)
cv2.waitKey(0)

