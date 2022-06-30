import os
import cv2
import numpy as np

n = 107
for i in range(n):
    os.system('labelme_json_to_dataset labeled_green/json/%d.json -o labeled_green/data/%d_json'%(i,i))

train_image = 'labeled_green/train_image/'
train_label = 'labeled_green/train_label/'
 
for i in range(n):
    print(i)
    img=cv2.imread('labeled_green/data/%d_json/img.png'%i)
    label=cv2.imread('labeled_green/data/%d_json/label.png'%i)
    print(img.shape)
    label=label/np.max(label[:,:,2])*255
    label[:,:,0]=label[:,:,1]=label[:,:,2]
    print(np.max(label[:,:,2]))
    # cv2.imshow('l',label)
    # cv2.waitKey(0)
    print(set(label.ravel()))
    cv2.imwrite(train_image+'%d.png'%i,img)
    cv2.imwrite(train_label+'%d.png'%i,label)
