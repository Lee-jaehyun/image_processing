import cv2
import os
import numpy as np
import pytesseract
import json
from collections import OrderedDict
from deskew import determine_skew
from angle4 import rotate

img = cv2.imread("../../Desktop/unnamed.png")

cv2.imshow("img", img)
cv2.waitKey(0)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#th, src_bin = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)

#src_bin = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

cv2.imshow("img", grayscale)
cv2.waitKey(0)
'''
image = np.array(img)

w, h = image.shape[:2]

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#angle = determine_skew(grayscale)
angle = determine_skew(grayscale[int(h/2):, int(w/3):int(w*2/3)])

print(angle)

if (angle < -70) or (angle > 80):
    rotated = image
else:
    rotated = rotate(image, angle, (0, 0, 0))
 #   x, y = 500, 800         # x, y = 1000, 800
  #  h, w = 1000, 1200        #  h, w = 3000, 2200
   # rotated = rotated[x:x +  h, y:y + w].copy()

#rotated = cv2.resize(rotated, None, fx=2.0, fy=2.0)   #, interpolation=cv2.INTER_CUBIC)

cv2.imshow('rotated', rotated)
cv2.waitKey(0)

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)

mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)

clone = rotated.copy()

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(clone, hulls, 1, (0,255,0))

cv2.imshow('img', clone)
cv2.waitKey(0)

mask = np.zeros((rotated.shape[0], rotated.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255,255,255), -1)

text_only = cv2.bitwise_and(rotated, rotated, mask=mask)
cv2.imshow("text_only", text_only)
cv2.waitKey(0)

output = []

grayscale = cv2.cvtColor(text_only, cv2.COLOR_BGR2GRAY)
th, src_bin = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)
'''

output= []
custom_config = r'--oem 1 --psm 3'
text = pytesseract.image_to_string(grayscale, lang="letsgodigital")
#text = pytesseract.image_to_string(rotated[y:y + h, x:x + w], lang=None, config=custom_config)
output.append(text.split())

file_data = OrderedDict()
file_data['text'] = output

output_data = json.dumps(file_data, ensure_ascii=False, indent="\t")
print(output_data)
