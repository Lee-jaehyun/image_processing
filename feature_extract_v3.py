import cv2
import numpy as np
from mser2 import mser_process


img = cv2.imread('../374.png')
#img = cv2.imread('../../../../Volumes/USB/data/train/혈압계/k009.jpg')
#img = cv2.imread("../cnn/output1.jpg")
#img = cv2.imread('../../../../Volumes/USB/data/처방전/j008.jpg')
#img = cv2.imread('../../../../Volumes/USB/data/train/체온계/j001.jpg')
#img = cv2.imread('../../../../Volumes/USB/data/train/혈당계/j008.jpg')

def img_process(img):
    #gray scale로 이미지 변환
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # ( 6,6 )
  #  img = clahe.apply(grayscale)

    blurred = cv2.GaussianBlur(grayscale, (5,5), 0)


    #edged = cv2.Canny(blurred, 100, 255, 255)

    #edged = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, 3)

    #th, src_bin = cv2.threshold(blurred, 120, 200, cv2.THRESH_BINARY)   # 150      # 80

    #src_bin = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    #src_bin = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    return blurred

def canny_edge(img):
    src = cv2.Canny(img, 50, 140, 255)      # 70, 160
    #src = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    #ret3, src = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #src = src_bin
    kernel = np.ones((3,1), np.uint8)
    dilation = cv2.dilate(src, kernel, iterations=3)

    cv2.imshow('dilation', dilation)
    cv2.waitKey(0)

    kernel_size = (5, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dst = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    return dst


def find_contours(src):

    cv2.imshow('src_bin', src)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        print("Failed to find Contours")
    else:
        pass

   # print(contours)

#cv2.imshow('contour', con_img)
#cv2.waitKey(0)

    mask = np.zeros(src.shape, dtype=np.uint8)

    test_image_list = []

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
    #    print(x, y, w, h)
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    # 바운딩 박스 크기 조절하는 부분! (조건)
        if r > 0.3 and w > 30 and h > 30:
            cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            cv2.imshow('sebun', img[y:y + h, x:x + w])
            cv2.waitKey(0)
           # test_image = mser_process(img[y:y + h, x:x + w])
            test_image = img[y:y + h, x:x + w]
            test_image_list.append(test_image)
            #cv2.imshow('test_img', test_image)
            #cv2.waitKey(0)

    if len(test_image_list) == 0:
        print("Failed to find Contours")
        test_image = None

    return test_image



def main():
    image = img_process(img)
    s_image = canny_edge(image)
    find_contours(s_image)


if __name__ == "__main__":
    main()