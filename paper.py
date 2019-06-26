'''
For paper publication
This basic code works well for text segmentation for images
Input for this code are entite images containing text and segment

to experiment with sliding window new copy of it created
encoder-decoder-train_5.py

to experiment with architecure newly created copy is
encoder-decoder-train_6.py

'''


import numpy as np
np.random.seed(1000) # for reproducibility
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D,UpSampling2D
from keras.layers import Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import *
from keras.models import load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os
import cv2
import sys
from matplotlib import pyplot as plt
cwd=os.getcwd()+"//"
#oldFiles=os.listdir(cwd+"results2//")


'''
this code is modified for new segmentaion
'''

'''
    0) dont change older path
path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//"
'''
pathOld="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//"
'''
    1)publication experiment path
'''
path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//publicationData//"
expN0=8
#pathOld=path+str(expN0)+"//imageWord//"

print("\n\t experiment no=",expN0)
'''
1
path_x = path+'/images2/'
path_y = path+'/segment2/'
'''
#0 wights model is (paper_results.hdf5)
#path_x = path+'/imageWord/' #only hands
#path_y = path+'/segWord/' #segmented data

'''
    2) EXPERIMENTS FOR PAPER
'''
path_x = path+str(expN0)+'//train/imageWord/' #only hands
path_y = path+str(expN0)+'//train/segWord/' #segmented data
'''
'''
total = 0

dump= os.listdir(path_x)
dumpLen=500 #len(dump)
print("\n\t dumpLen1=",dumpLen)

'''
dump=os.listdir(path_y)
dumpLen= len(dump)
print("\n\t dumpLen2=",dumpLen)
'''
maxImageProcess= dumpLen
#for pos in range(len(path_x)):

noException=0
blackOnWhite=0
bothBlackAndWhite=0 # considers black and white images by inverting

insize=256
X_train=np.zeros((maxImageProcess,insize,insize))
y_train=np.zeros((maxImageProcess,insize,insize))


for indxImg,img in enumerate(sorted(dump)):

    if indxImg %100==0:
        print "\n\tindxImg=",indxImg,"\t dumpLen=",dumpLen
    if indxImg>=maxImageProcess:
            break
    try:
        originalIm = cv2.imread(path_x+img)

        originalIm = cv2.cvtColor(originalIm, cv2.COLOR_BGR2GRAY)
        #originalIm=np.expand_dims(originalIm,0)
        #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//images1//"+img,originalIm)

        # img=img.split(".framed")[0]
        # img=img+".lines.png"

        # img=img.split(".png")[0]
        # img=img+".lines.png"


        # print("\n\t img=",img)
        #
        # print("\n\t img=", img, "\t ", os.path.isfile(path_y + img), "\t ", os.path.isdir(path_y))
        # print("\n\t isFile=",os.path.isfile(path_y+img))


        segmentedIm = cv2.imread(path_y+img)
        segmentedIm = cv2.cvtColor(segmentedIm, cv2.COLOR_BGR2GRAY)
        #segmentedIm=np.expand_dims(segmentedIm,0)


        # print "\n\t 1indxImg=",indxImg,"\t image shape=",segmentedIm.shape
        # print "\n\t 2indxImg=",indxImg,"\t image shape=",originalIm.shape


        #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//segment1//"+img,segmentedIm)

        X_train[indxImg] = cv2.resize(originalIm, (insize, insize)) #originalIm
        y_train[indxImg] = cv2.resize(segmentedIm, (insize, insize))

    except Exception as e:
        noException+=1
        print "\n\t e=",e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no", exc_tb.tb_lineno)
        #input("check exception")


print "\n\t noException=",noException

X_train=X_train.reshape([-1,insize, insize,1])
y_train=y_train.reshape([-1,insize, insize,1])

tests = os.listdir(pathOld+'/test/')#["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
noTestImages=len(tests)
print "\n\t noTestImages=",noTestImages


X_test = np.zeros((noTestImages,insize, insize))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
testException=0

for pos in range(len(tests)):
    try:

        temp=cv2.imread(pathOld+'/test/'+tests[pos])
        #print "\n\t test size",temp.shape
        #showImage(str(pos),temp)

        im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        #ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if blackOnWhite == 1:
            temp = (255 - temp)

        #X_test[pos] = cv2.resize(temp,(128,128))
        X_test[pos] = cv2.resize(im,(insize, insize))
        X_test1.append(temp)

        # if 1:#bothBlackAndWhite==1:
        #     X_test1.append(255-temp)
    except Exception as e:
        print "\n\t file name =",tests[pos]
        testException+=1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no in test images=", exc_tb.tb_lineno)

X_test=X_test.reshape([-1,insize, insize,1])

print "\n\t testException=",testException

X_train-=128.0
X_train/=128.0
y_train-=128.0
y_train/=128.0
X_test-=128.0
X_test/=128.0

print "1.X_train shape=",X_train[0].shape
print "2.y_train shape=",X_train.shape
print "3.X_test shape=",X_test.shape


def createModel():

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    clf = Sequential()

    clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(insize, insize,1), padding='same'))
    #clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(None,None,1), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))

    clf.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))

    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(1,1)))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))

    clf.add(Convolution2D(filters=512*2,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    #clf.add(MaxPooling2D(pool_size=(2,2),, strides=(1,1))
    # comment down

    clf.add(UpSampling2D((2,2)))
    clf.add(Convolution2D(filters=256*2,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    #writeName = "fusion_" + str(j) + "_" + str(i) + "_" + str(hitIndx)  # this is image name

    clf.add(UpSampling2D((2,2)))
    clf.add(Convolution2D(filters=128,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(UpSampling2D((2,2)))
    clf.add(Convolution2D(filters=64,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(1, (3, 3), padding='same'))
    clf.add(Activation('tanh'))

    clf.compile(optimizer='adam',loss='mse',metrics=['mae'])
    #clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return clf


def get_callbacks(filepath, patience):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    msave =ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1)

    return [es, msave]

''' 
    BEFORE EXPERIMT MODEL PATH WAS HERE
'''
#file_path = cwd+"//models//paper_model_weights_newData.hdf5"

'''
    3) model weight path
'''
file_path = path+str(expN0)+"//model//paper_results_"+str(expN0)+"_"+str(dumpLen)+"_"+"_.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=20)

clf=createModel()

#print(clf.summary())
model_json=clf.to_json()

'''
with open(cwd+"papermodelArch.json", "w") as json_file:
    json_file.write(model_json)
'''

'''
    4) Save arcitecture
'''
with open(path+str(expN0)+"//architecture//papermodelArch.json", "w") as json_file:
    json_file.write(model_json)

#print clf.summary()

#keras.callbacks.ModelCheckpoint(cwd+'//models//', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

clf.fit(X_train,y_train,batch_size=15, epochs=50,validation_split=0.2,callbacks=callbacks,shuffle=True,verbose=2)
#clf.save(cwd+'//models//model-10.h5')
clf.save(file_path)

sys.stdout.flush()
y_out = clf.predict(X_test)


y_out*=128.0
y_out+=128.0

'''
    5)RESULTS PATH
'''
'''
for y in range(y_out.shape[0]):
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (w,h)) #originalIm
    nm=tests[y]
    cv2.imwrite(path+"//modelPrediction//"+nm+'t.jpg',X_test1[y])
    cv2.imwrite(path+"//modelPrediction//"+nm+'s1gray.jpg',tmp)
'''

for y in range(y_out.shape[0]):
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (w,h)) #originalIm
    nm=tests[y]
    cv2.imwrite(path+str(expN0)+"//modelPrediction//"+nm+'t.jpg',X_test1[y])
    cv2.imwrite(path+str(expN0)+"//modelPrediction//"+nm+'s1gray.jpg',tmp)
