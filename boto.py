import boto3
from PIL import Image
import numpy as np
import cv2
from botocore.client import Config
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import datetime
import urllib.request


ACCESS_KEY_ID = "AKIA277UZJGM5XEGBXGA"
ACCESS_SECRET_KEY = "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"
region = 'ap-northeast-2'
bucket_name = 'dataset-dhealth'
prefix ='food_data/train/fruit/fruit_salad'


s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=ACCESS_SECRET_KEY, region_name=region)

'''
obj_list = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
contents_list = obj_list['Contents']
print(contents_list[0]['Key'])

img = cv2.imread(contents_list[0]['Key'])
print(img)'''








#file path ==> object

train_data = []


paginator = s3.get_paginator('list_objects_v2')

response_iterator = paginator.paginate(
    Bucket=bucket_name,
    Prefix=prefix
)

key = []

for page in response_iterator:
   # print(page)
    for content in page['Contents']:
        key.append(content['Key'])
        print(content['Key'])

# time stamp
current = datetime.datetime.now()

#for i in range(100):      #len(key)

#   response = s3.get_object(Bucket='dataset-dhealth', Key=key[i])      #Key='food_data/train/vegetable/sangchu_salad/B140731XX_00867.jpg'
#print(response)
#data = response['Body'].get()
#print(data)


#   file_stream = response['Body']

#   im = Image.open(file_stream)
#   train_data.append(np.array(im))

#print(np.array(im).shape)

  # cv2.imshow('wow', np.array(im))
  # cv2.waitKey(0)


#print(len(train_data))

#cv2.imshow('wow', train_data[5])
#cv2.waitKey(0)


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


for i in range(1000):
    img = url_to_image('https://dataset-dhealth.s3.ap-northeast-2.amazonaws.com/' + key[i])
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    train_data.append(img)


endpoint = datetime.datetime.now()
print(endpoint - current)
print(len(train_data))

cv2.imshow('train', train_data[10])
cv2.waitKey(0)



#img = url_to_image('https://dataset-dhealth.s3.ap-northeast-2.amazonaws.com/food_data/train/cereals/dduck/A240210XX_00866.jpg')
