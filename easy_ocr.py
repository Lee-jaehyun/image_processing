import easyocr
import time
import datetime
import cv2
from deskew import determine_skew
from angle4 import rotate


reader = easyocr.Reader(['ko'], gpu=False)
image = cv2.imread("../../Desktop/image3.jpeg")
image = cv2.resize(image, dsize=(960, 1280), interpolation=cv2.INTER_AREA)

#w, h = image.shape[:2]

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#angle = determine_skew(grayscale)
angle = determine_skew(grayscale)    #[int(h/3):, int(w/4):int(w*2/3)]
print(angle)

if (angle < -78) or (angle > 80):
    rotated = image
else:
    rotated = rotate(image, angle, (0, 0, 0))

#rotated = cv2.resize(rotated, dsize=(800, 960), interpolation=cv2.INTER_AREA)

cv2.imshow("rotated", rotated)
cv2.waitKey(0)

start = time.time()
result = reader.readtext(rotated)

end_ocr = time.time()

for i in result:
    print(i[1])

sec = (end_ocr - start)
print("TOTAL_process :", datetime.timedelta(seconds=sec))