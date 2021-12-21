import torch
import cv2
from ycbcr import homomorphic_filter


def sort_values(output_data):
    output_data["sort_v"] = output_data["xmin"] + (output_data["ymin"]+(output_data["ymax"]-output_data["ymin"])*0.85)
    sorted_ctrs = output_data.sort_values(by=["sort_v"])
    return sorted_ctrs



#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolo_road_det15/weights/best.pt', force_reload=True)

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

#model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

#torch.save(model, "./third_seven.pt")

model = torch.load("third_seven.pt")

model.eval()

#img = cv2.imread('datasets/realroad/images/test/road113.png')
#img = cv2.imread('../IMG_0818.jpeg')

path = '../test_data_set2/IMG_1134.jpg'
#img = homomorphic_filter(path)

#cv2.imshow("img", img)
#cv2.waitKey(0)

img = cv2.imread(path)

#ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("gray", img)
#cv2.waitKey(0)

#y2, cr2, cb2 = cv2.split(ycbcr)

#new_img = (cr2/2.0) + (cb2/2.0)

#cv2.imshow('cr', img)
#cv2.imshow('new', new_img)
#cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
results = model(img, size=640)    #size=640
#print(results)

crops = results.crop(save=False)
#print(type(crops[0]))

yolo_result = crops

print(len(yolo_result))

for i in range(len(yolo_result)):
    print(yolo_result[i]["label"])

output_data = results.pandas().xyxy[0]
print(results.pandas().xyxy[0]["ymin"])

print(output_data.sort_values(by=["xmin"]))


aa = sort_values(output_data)
print(aa)
print(list(aa["class"]))

results.print()
results.show()