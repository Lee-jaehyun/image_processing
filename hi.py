import cv2
import numpy as np
import urllib.request

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


img = url_to_image('https://dataset-dhealth.s3.ap-northeast-2.amazonaws.com/food_data/train/drink1/boicha/A200214XX_31296.jpg')

cv2.imshow('img', img)
cv2.waitKey(0)