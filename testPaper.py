from keras.models import load_model
import os
import cv2
import numpy as np
import sys

#clf.save('model-10.h5')
cwd=os.getcwd()
#file_path = cwd+"//models//paper_model_weights_newData.hdf5" #
#file_path = cwd+"//models//paper_model_weights.hdf5" #
#file_path = cwd+"//models//paper_model_weights.hdf5"

'''
    CHANGE 0) SPECIFY EXPERIMENT NO, IT HELPS TO DECIDES FOLEDER PATH
'''
exp=6
'''
    change 1) Location and name of model
'''
#file_path = cwd+"//models//paper_results.hdf5"

##################################################################################################
'''
    # experiment 1 model
'''

path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/"+str(exp)+"//"
modelName="paper_results_5_5000__.hdf5"# _500_ indicates training data images
file_path = os.path.join(path,"model",modelName)

##################################################################################################

clf = load_model(file_path)
'''
test_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/test2//"
test_save_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/temp//"
'''

'''
test_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/paperTest//"
test_save_folder = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/paperResults//"
'''


'''
    change 2) change test image and result store locations
'''

test_folder = os.path.join(path,"test","imageWord")
test_save_folder = os.path.join(path,"results")+"//"

print("\n\t test_save_folder=",test_save_folder,"\t is exist=",os.path.isdir(test_save_folder))
length = len(os.listdir(test_folder))
print("\n\t no of test images=",length)
inSize=256
X = np.zeros((length,inSize,inSize))
read=[]
total=0
blackOnWhite=1

for img in os.listdir(test_folder):

    temp = cv2.imread(os.path.join(test_folder,img))
    # print("\n\t test_folder+img=",test_folder+img)
    # print("\n\t temp shape=",temp.shape)
    im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if blackOnWhite == 1:
        #temp = (255 - temp)
        im = (255 - im)

    X[total] = cv2.resize(im, (inSize,inSize))
    total+=1
    read.append(img)
    cv2.imwrite(test_save_folder +img+"o"+".png",temp)

X=X.reshape([-1,inSize, inSize,1])

X-=128.0
X/=128.0
y_out = clf.predict(X)
y_out*=128.0
y_out+=128.0

X*=128.0
X+=128.0

'''
    binarization threshold
    before publication results threshold is threshold=150
'''

threshold=100
for y in range(len(read)):
    if y%10==0:
        print("\n\t index=",y)
    #print(read[y],"\t indx=",y)
    pred=y_out[y]
    pred=cv2.resize(pred,(2500,3300))

    pixelDown=np.where(pred<threshold)
    pred[pixelDown]=0

    pixelUP=np.where(pred>=threshold)
    pred[pixelUP]=255

    #savePath=os.path.join(test_save_folder,read[y],"UP.jpg")
    #print("\n\t path=",test_save_folder+"//"+read[y]+"UP.jpg")
    cv2.imwrite(test_save_folder+"//"+read[y]+"UP.jpg",pred)
    #inImage=X[y]
    #inImage=cv2.resize(inImage,(500,500))
    #cv2.imwrite(test_save_folder + read[y]+"o"+".png",inImage)
