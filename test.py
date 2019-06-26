from keras.models import load_model
import os
import cv2
import numpy as np
import sys

#clf.save('model-10.h5')
cwd=os.getcwd()
file_path = cwd+"//models//model_weights.hdf5"
clf = load_model(file_path)
test_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/test//"
test_save_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/temp//"
length = len(os.listdir(test_folder))
X = np.zeros((length,128,128,3))
read=[]
total=0
blackOnWhite=0

for img in os.listdir(test_folder):

    temp = cv2.imread(test_folder+img)

    im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if blackOnWhite == 1:
        temp = (255 - temp)

    X[total] = cv2.resize(temp, (128, 128))
    total+=1
    read.append(img)

X-=128.0
X/=128.0
y_out = clf.predict(X)
y_out*=128.0
y_out+=128.0



X*=128.0
X+=128.0

for y in range(len(read)):
    print(read[y],"\t indx=",y)
    cv2.imwrite(test_save_folder+read[y],y_out[y])
    cv2.imwrite(test_save_folder +"ori_" +read[y], X[y])
