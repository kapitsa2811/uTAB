'''
For paper publication

network is borrowed from paper.py except in out is 3 channel
Input for this code are entite images containing text and segment

to experiment with sliding window new copy of it created
encoder-decoder-train_5.py

to experiment with architecure newly created copy is
encoder-decoder-train_6.py

'''


import numpy as np
np.random.seed(1000) # for reproducibility
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import MaxPooling2D,UpSampling2D
from keras.layers import Dropout,Dense,Flatten,BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import concatenate
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import merge
from keras.layers import Reshape,Lambda, Dense, Bidirectional, GRU, Flatten, TimeDistributed, Permute, Activation, Input
from keras.layers import LSTM
from keras.optimizers import *
from keras.models import load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers.convolutional_recurrent import ConvLSTM2D
import os
import cv2
import sys
#from paper.networkExperiments.networks import *
#from pu networkExperiment.networks import *

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
pathOld=".//publicationData//"
'''
    1)publication experiment path
'''
#path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//publicationData//"
path=".//publicationData//"
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
#path_x = path+str(expN0)+'//train/imageWord/' #only hands
path_x = path+str(expN0)+'//train/blocks/' #
path_x1 = path+str(expN0)+'//train/contourBlocks/' #

path_y = path+str(expN0)+'//train/segWord/' #segmented data
#path_y = path+str(expN0)+'//train/segWord/' #segmented data
'''
'''
total = 0

dump= os.listdir(path_x)
dumpLen= 480#len(dump)
print("\n\t dumpLen1=",dumpLen)
print("\n\t dumpLen1=",len(os.listdir(path_x)))


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
unableKernel=0
unableContourMaps=1

insize=256
X_train=np.zeros((maxImageProcess,insize,insize,3))
y_train=np.zeros((maxImageProcess,insize,insize,3))



for indxImg,img in enumerate(sorted(dump)):

    if indxImg %100==0:
        print ("\n\tindxImg=",indxImg,"\t dumpLen=",dumpLen)
    if indxImg>=maxImageProcess:
            break
    try:
        originalIm = cv2.imread(path_x+img)
        originalIm1 = cv2.imread(path_x1+img)
        row,col,ch=originalIm.shape

        # print(originalIm.shape)
        # print(originalIm1.shape)


        if unableKernel==1:
            for i in range(row):
                for j in range(col):
                    originalIm[i,j,1]=(i*1.0/col*1.0)*1.0
                    originalIm[i, j, 2] = (j*1.0/row*1.0)*1.0


        if unableContourMaps==1:
            delMeTemp=originalIm1[:,:,0]

            #print("\n\t shape=",delMeTemp.shape)

            originalIm[:,:,1]=0.3*delMeTemp
            #originalIm = cv2.addWeighted(originalIm, 0.7, originalIm1, 0.3, 0)

            #for i in range(row):
                # for j in range(col):
                #     originalIm[i,j,0]=delMeTemp[i,j]
                #     #originalIm[i, j, 0] = (j*1.0/row*1.0)*1.0
                #

            #cv2.imwrite("/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/train//delMe//"+img,originalIm)

            # plt.imshow("fusinImage",originalIm)
            # plt.waitforbuttonpress()
            '''
            for i in range(row):
                for j in range(col):
                    originalIm[i,j,1]=(i*1.0/col*1.0)*1.0
                    originalIm[i, j, 2] = (j*1.0/row*1.0)*1.0
                '''

        #originalIm = cv2.cvtColor(originalIm, cv2.COLOR_BGR2GRAY)
        #originalIm=np.expand_dims(originalIm,0)
        #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//images1//"+img,originalIm)

        # img=img.split(".framed")[0]
        # img=img+".lines.png"

        # img=img.split(".png")[0]
        # img=img+".lines.png"


        # print("\n\t img=",img)
        #
        # print("\n\t img=", img, "\t ", os.path.isfile(path_y + img), "\t ", os.path.isdir(path_y))
        #print("\n\t isFile=",os.path.isfile(path_y+img))


        segmentedIm = cv2.imread(path_y+img)
        #segmentedIm = cv2.cvtColor(segmentedIm, cv2.COLOR_BGR2GRAY)
        #segmentedIm=np.expand_dims(segmentedIm,0)


        # print "\n\t 1indxImg=",indxImg,"\t image shape=",segmentedIm.shape
        # print "\n\t 2indxImg=",indxImg,"\t image shape=",originalIm.shape


        #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//segment1//"+img,segmentedIm)

        X_train[indxImg] = cv2.resize(originalIm, (insize, insize)) #originalIm
        y_train[indxImg] = cv2.resize(segmentedIm, (insize, insize))

    except Exception as e:
        noException+=1
        print ("\n\t e=",e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        #print("\n\t segmentedIm size=",segmentedIm.shape)
        print("\n\t line no", exc_tb.tb_lineno)
        #input("check exception")


print ("\n\t noException=",noException)



# X_train=X_train.reshape([-1,insize, insize,3])
# y_train=y_train.reshape([-1,insize, insize,3])

tests = os.listdir(pathOld+'/test/')#["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
noTestImages=len(tests)
print ("\n\t noTestImages=",noTestImages)


X_test = np.zeros((noTestImages,insize, insize,3))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
testException=0

contourTestPaths="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/train/contourBlocksTest//"

for pos in range(len(tests)):
    try:

        temp=cv2.imread(pathOld+'/test/'+tests[pos])
        temp1=cv2.imread(contourTestPaths+tests[pos])


        if unableContourMaps==1:
            delMeTemp=temp1[:,:,0]

            #print("\n\t shape=",delMeTemp.shape)

            temp[:,:,1]=0.3*delMeTemp


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

#X_test=X_test.reshape([-1,insize, insize,1])

print("\n\t testException=",testException)

X_train-=128.0
X_train/=128.0
y_train-=128.0
y_train/=128.0
X_test-=128.0
X_test/=128.0

print ("1.X_train shape=",X_train[0].shape)
print ("2.y_train shape=",X_train.shape)
print ("3.X_test shape=",X_test.shape)


def createModel():

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    clf = Sequential()

    #model = ResNet50()
    #model = Model(inputs=model.input, outputs=model.get_layer("activation_40").output)

    #clf = model.output

    clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(insize, insize,3), padding='same'))
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

    '''
        RNN efforts
    
    clf.add(Permute((2, 3, 1), name='permute'))
    clf.add(TimeDistributed(Flatten(), name='for_flatten_by_name'))

    # RNN part
    clf.add(Bidirectional(LSTM(256, return_sequences=True), name='BLSTM1'))
    #clf.add(BatchNormalization(name='rnn_BN'))
    #clf.add(Bidirectional(LSTM(256, return_sequences=True), name='BLSTM2'))
    '''
    '''
    '''
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

    clf.add(Convolution2D(3, (3, 3), padding='same'))
    clf.add(Activation('tanh'))

    clf.compile(optimizer='adam',loss='mse',metrics=['mae'])
    #clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return clf


def createModel3():#https://github.com/keras-team/keras/issues/6200
    input_shape = (300, 300, 3)
    conv_filters = 16
    kernel_size = (3, 3)

    seq = Sequential()

    inp_layer = Input(name='the_input', shape=input_shape, dtype='float32', batch_shape=(1, 300, 300, 3))
    # x = BatchNormalization()(inp_layer)

    x = Convolution2D(1024, kernel_size, padding='same',
               activation='tanh', kernel_initializer='he_normal',
               name='conv1')(inp_layer)
    x = Convolution2D(512, kernel_size, padding='same',
               activation='tanh', kernel_initializer='he_normal',
               name='conv1_1')(x)
    x = Convolution2D(32, kernel_size, padding='same',
               activation='tanh', kernel_initializer='he_normal',
               name='conv1_2')(x)

    '''
    # x = Conv2D(16, kernel_size, padding='same',
    #                    activation='tanh', kernel_initializer='he_normal',
    #                     name='conv2')(x)
    conv_to_LSTM_dims = (1, 300, 300, 32)
    x = Reshape(target_shape=conv_to_LSTM_dims, name='reshapeconvtolstm')(x)

    x = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                   input_shape=(None, 300, 300, 32),
                   padding='same', return_sequences=True, stateful=True)(x))

    LSTM_to_conv_dims = (300, 300, 8)
    x = Reshape(target_shape=LSTM_to_conv_dims, name='reshapelstmtoconv')(x)
    '''

    x = Convolution2D(3, (1, 1), padding='same',
               activation='sigmoid', kernel_initializer='he_normal',
               name='decoder')(x)

    clf = Model(inputs=inp_layer, outputs=x)
    clf.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae'])
    return clf


def createModel1():



    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #clf = Sequential()

    #inputs = Input(shape=(384,384,3))
    inputs = Input(shape=(insize, insize,3))
    y=Convolution2D(filters=3,kernel_size=(2,2),input_shape=(insize,insize,3),padding='same')(inputs)
    y1=BatchNormalization()(y)
    y2=Activation('relu')(y)
    y3=MaxPooling2D(pool_size=(2,2))(y2)

    #1# clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(insize, insize,3), padding='same'))
    #clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(None,None,1), padding='same'))

    #2#clf.add(BatchNormalization())
    #3#clf.add(Activation('relu'))
    #4#clf.add(MaxPooling2D(pool_size=(2,2)))

    '''
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
    '''


    y33=UpSampling2D((2,2))(y3)
    y22=Convolution2D(filters=3,kernel_size=(3,3), padding='same')(y33)
    y22=BatchNormalization()(y22)
    y11=Activation('relu')(y22)
    y111 = Convolution2D(filters=3, kernel_size=(3, 3), padding='same')(y11)
    #y111 = Convolution2D(filters=3, kernel_size=(3, 3), padding='same')(y11)
    #con=Concatenate([y2,y22])
    out=concatenate([y111,y],axis=0)
    #out =merge([y11,y], mode='concat', concat_axis=1)
    yout=Activation('tanh')(out)

    print("")

    #5#clf.add(UpSampling2D((2,2)))
    #6#clf.add(Convolution2D(filters=64,kernel_size=(3,3), padding='same'))
    #7#clf.add(BatchNormalization())
    #8#clf.add(Activation('relu'))
    #8#clf.add(Activation('relu'))

    #9#clf.add(Convolution2D(3, (3, 3), padding='same'))
    #10#clf.add(Activation('tanh'))

    #11#clf.compile(optimizer='adam',loss='mse',metrics=['mae'])
    #clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])
    clf=Model(input=inputs,output=yout)

    return clf


def createModel2(pretrained_weights=None, input_size=(256, 256, 1)):

    inputs = Input(shape=(insize, insize,3))
    conv1 = Convolution2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Convolution2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

    conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

    conv4 = Convolution2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Convolution2D(1024, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Convolution2D(1024, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Convolution2D(512, kernel_size=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Convolution2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Convolution2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Convolution2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Convolution2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Convolution2D(128, kernel_size=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Convolution2D(64, kernel_size=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Convolution2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Convolution2D(2, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Convolution2D(3, kernel_size=(1,1), activation='tanh')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae'])

    # model.summary()

    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)

    return model

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
callbacks = get_callbacks(filepath=file_path, patience=15)

clf=createModel2()
#clf.compile(optimizer='adam',loss='mse',metrics=['mae'])
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

print("\n\t X_test1",len(X_test1))

clf.fit(X_train,y_train,batch_size=10, epochs=1000,validation_split=0.2,callbacks=callbacks,shuffle=True,verbose=2)

modelpath="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/model//"
clf.save(modelpath+'///model-10.h5')
#clf.save(file_path)

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
