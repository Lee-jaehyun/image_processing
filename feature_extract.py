import cv2
import numpy as np

img = cv2.imread('omron.jpeg')

#gray scale로 이지미 변환
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, src_bin = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)

src = cv2.Canny(src_bin, 120, 200)

cv2.imshow('src_bin', src)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(contours)
#cv2.imshow('contour', con_img)
#cv2.waitKey(0)

mask = np.zeros(src_bin.shape, dtype=np.uint8)

con_img = []
con_img.append([])

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    print(x, y, w, h)
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    if r > 0.5 and w > 10 and h > 10:
        cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        cv2.imshow('sebun', img[y:y + h, x:x + w])
        cv2.waitKey(0)
#        con_img[idx].append(x)
#        con_img[idx].append(y)
#        con_img[idx].append(w)
#        con_img[idx].append(h)



# show image with contours rect
cv2.imshow('rects', img)
cv2.waitKey(0)

print(con_img[0])

cv2.imshow('con1', img[100:400, 300: 600])
cv2.waitKey(0)

