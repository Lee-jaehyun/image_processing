import cv2
import numpy as np
import pytesseract
import json
from collections import OrderedDict
from deskew import determine_skew
from angle4 import rotate

from PIL import Image
from io import BytesIO




img = cv2.imread("../../Desktop/skewed1.jpeg")

#img = Image.open(data)

image = np.array(img)
cv2.imshow('raw', img)
cv2.waitKey(0)

#image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

w, h = image.shape[:2]

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#angle = determine_skew(grayscale)
angle = determine_skew(grayscale[int(h/2):, :])

print(angle)

if (angle < -80) or (angle > 80):
    rotated = image
else:
    rotated = rotate(image, angle, (0, 0, 0))
 #   x, y = 500, 800         # x, y = 1000, 800
  #  h, w = 1000, 1200        #  h, w = 3000, 2200
   # rotated = rotated[x:x + h, y:y + w].copy()

#rotated = cv2.resize(rotated, None, fx=2.0, fy=2.0)   #, interpolation=cv2.INTER_CUBIC)

cv2.imshow('rotated', rotated)
cv2.waitKey(0)

grayscale = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

#th, src_bin = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)

#src_bin = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)
src_bin = cv2.Canny(grayscale, 80, 160)

cv2.imshow('src_bin', src_bin)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#print(contours)
#cv2.imshow('contour', con_img)
#cv2.waitKey(0)

mask = np.zeros(src_bin.shape, dtype=np.uint8)

output = []

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    if r > 0.2 and w > 50 and h > 50:
        cv2.rectangle(rotated, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

        test_img = rotated[y:y+h, x:x+w]
        print(test_img.shape)
        cv2.imshow('teest',test_img)
        cv2.waitKey(0)


        #cv2.imshow('sebun', rotated[y:y + h, x:x + w])
       # cv2.imshow('test_img', rotated_2)
       # cv2.waitKey(0)
        test_img2 = cv2.resize(test_img, None, fx=2, fy=2)

        custom_config = r'--oem 1 --psm 3'
        text = pytesseract.image_to_string(test_img2 , lang=None, config=custom_config)
        #text = pytesseract.image_to_string(rotated[y:y + h, x:x + w], lang=None, config=custom_config)
        output.append(text.split())

file_data = OrderedDict()
file_data['text'] = output

output_data = json.dumps(file_data, ensure_ascii=False, indent="\t")

print(output_data)
print(' ========================= ')
print(angle)


'''
heightImage, weightImage, _ = rotated.shape

cong = r"--oem 3 --psm 6 ouputbase digits"
boxes = pytesseract.image_to_boxes(rotated, config=cong)


## Forming bounding box around digits ##
for b in boxes.splitlines():
    b = b.split(" ")
    print(b)

    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    box = cv2.rectangle(rotated, (x, heightImage-y+1), (w, heightImage-h+1), (0, 0, 255), 3)
    cv2.putText(image, b[0], (x,heightImage-y+25), cv2.FONT_HERSHEY_COMPLEX,1,(20,30,255), 2)

cv2.imshow("detecting digits", box)
cv2.waitKey(0)
'''