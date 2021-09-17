import cv2
import numpy as np


#img = cv2.imread('../tttest.png')
#img = cv2.imread('../../../../Volumes/USB/data/train/혈압계/k003.jpg')
#img = cv2.imread('../../../../Volumes/USB/data/train/체중계/j003.jpg')
img = cv2.imread('../../../../Volumes/USB/data/train/체온계/j001.jpg')
#img = cv2.imread('../../../../Volumes/USB/data/train/혈당계/j001.jpg')
img = cv2.resize(img, (0,0), fx=1.5 , fy=1.5)


def img_process(img):
    #gray scale로 이미지 변환
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayscale, (5,5), 0)

    #edged = cv2.Canny(blurred, 100, 255, 255)

    #edged = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, 3)

    th, src_bin = cv2.threshold(blurred, 160, 200, cv2.THRESH_BINARY)   # 150      # 80

    #src_bin = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    #src_bin = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

    #blurred[100:200, 100:200] = 255

    return blurred

def preprocess(img, threshold, show=False, kernel_size=(5,5)):
    # 直方图局部均衡化
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))       #( 6,6 )
    img = clahe.apply(img)
    # 自适应阈值二值化

    #dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)      # 127

    #dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, threshold)
    # 闭运算开运算
   # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

    if show:
        cv2.imshow('equlizeHist', img)
        cv2.imshow('threshold', dst)
    return dst


def find_contours(src_bin):
    src = cv2.Canny(src_bin, 100, 180)      # 70, 160
    #src = src_bin
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(src, kernel, iterations=2)



    cv2.imshow('src_bin', dilation)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
        if r > 0.1 and w > 100 and h > 100:
            cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            cv2.imshow('sebun', src[y:y + h, x:x + w])
            cv2.waitKey(0)
            test_image = img[y:y + h, x:x + w]
            test_image_list.append(test_image)
            cv2.imshow('test_img', test_image)
            cv2.waitKey(0)

    if len(test_image_list) == 0:
        print("Failed to find Contours")
        test_image = None

    return test_image

#            test_image = cv2.resize(test_image, (200, 200))
#            test_image = np.asarray(test_image, dtype=np.float32)
 #           test_image = np.reshape(test_image, (1, 200 * 200))

            # th, test_image = cv2.threshold(test_image, 80, 255, cv2.THRESH_BINARY)

            ### Morphological Transformations 연산#####
         #   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
           # test_image = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel)


  #      _, y_val = model2.predict(test_image)
 #       print(y_val)




def main():
    process_img = img_process(img)
    dst = preprocess(process_img, 30)
    output = find_contours(dst)


if __name__ == '__main__':
    main()


# show image with contours rect
cv2.imshow('rects', img)
cv2.waitKey(0)