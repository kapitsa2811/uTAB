import keras
from keras.models import load_model
import os
import cv2
import numpy as np
import sys

path="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/model/paper_results_8_480__.hdf5"

imagePath="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/test//"

predPath="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/modelPrediction//"
model=load_model(path)


tests = os.listdir(imagePath)#["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
noTestImages=len(tests)
print ("\n\t noTestImages=",noTestImages)

insize=256
X_test = np.zeros((noTestImages,insize, insize,3))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
testException=0

X_test = np.zeros((noTestImages,insize, insize,3))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
testException=0
blackOnWhite=0

for pos in range(len(tests)):
    try:

        temp=cv2.imread(imagePath+tests[pos])
        #print "\n\t test size",temp.shape
        #showImage(str(pos),temp)

        im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        #ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im=temp
        if blackOnWhite == 1:
            temp = (255 - temp)

        #X_test[pos] = cv2.resize(temp,(128,128))
        X_test[pos] = cv2.resize(im,(insize, insize))
        X_test1.append(temp)

        # if 1:#bothBlackAndWhite==1:
        #     X_test1.append(255-temp)
    except Exception as e:
        print ("\n\t file name =",tests[pos])
        testException+=1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no in test images=", exc_tb.tb_lineno)


X_test-=128.0
X_test/=128.0

print ("3.X_test shape=",X_test.shape)



for pos in range(len(tests)):
    try:

        temp=cv2.imread(imagePath+tests[pos])
        #print "\n\t test size",temp.shape
        #showImage(str(pos),temp)

        im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        #ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im=temp
        if blackOnWhite == 1:
            temp = (255 - temp)

        #X_test[pos] = cv2.resize(temp,(128,128))
        X_test[pos] = cv2.resize(im,(insize, insize))
        X_test1.append(temp)

        # if 1:#bothBlackAndWhite==1:
        #     X_test1.append(255-temp)
    except Exception as e:
        print ("\n\t file name =",tests[pos])
        testException+=1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no in test images=", exc_tb.tb_lineno)



result=model.predict(X_test)

result*=128.0
result+=128.0


for indx,img in enumerate(os.listdir(imagePath)):
    #image=cv2.imread(os.path.join(imagePath,img))
    image=X_test[indx]
    cv2.imwrite(os.path.join(predPath,img),image)

    img1=img.split(".")[0]+"_pred_"+".jpg"
    cv2.imwrite(os.path.join(predPath,img1),result[indx])








