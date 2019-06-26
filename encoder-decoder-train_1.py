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

cwd=os.getcwd()+"//"
oldFiles=os.listdir(cwd+"results//")

for old in oldFiles:
    try:

        os.remove("/home/kapitsa/PycharmProjects/segmentation//Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/results/"+old)
    except Exception as e:
        print "\n\t cant delete=",old
        pass


'''
this code is modified for new segmentaion
'''

def showImage(name,image):
    print "\n\t image=",image.shape
    cv2.imshow(name,image)
    cv2.waitKey()

'''
angles = range(-2,3)
shifts = [[0,0],[0,1],[1,0],[1,1],[0,2],[2,0],[1,2],[2,1],[2,2],
                [0,-1],[-1,0],[-1,-1],[0,-2],[-2,0],[-1,-2],[-2,-1],[-2,-2],
                [1,-1],[1,-2],[2,-1],[2,-2],
                [-1,1],[-1,2],[-2,1],[-2,2]]
multiplier = len(angles)*len(shifts)
'''


# path_x = cwd+'/newData/X1/' #only hands
# path_y = cwd+'/newData/segment11/' #segmented data

# path_x = cwd+'/newData/image/' #only hands
# path_y = cwd+'/newData/segment/' #segmented data

path_x = cwd+'/newData/imageText1/' #only hands
path_y = cwd+'/newData/segmentText1/' #segmented data


total = 0

dump=os.listdir(path_x)
dumpLen=len(dump)
print("\n\t dumpLen1=",dumpLen)

dump=os.listdir(path_y)
dumpLen=len(dump)
print("\n\t dumpLen2=",dumpLen)

maxImageProcess=dumpLen
#for pos in range(len(path_x)):

noException=0
blackOnWhite=0

X_train=np.zeros((maxImageProcess,128,128,3))
y_train=np.zeros((maxImageProcess,128,128,3))

for indxImg,img in enumerate(sorted(dump)):

    print("\n\t img=",img,"\t ",os.path.isfile(path_x+img),"\t ",os.path.isdir(path_x))

    continue

    if indxImg %100==0:
        print "\n\tindxImg=",indxImg,"\t dumpLen=",dumpLen
        if indxImg>maxImageProcess:
            break
    try:
        originalIm = cv2.imread(path_x+img)
        #print "\n\t indxImg=",indxImg,"\t image shape=",originalIm.shape

        segmentedIm = cv2.imread(path_y+img)
        print("\n\t isFile=",os.path.isfile(path_y+img))
        print "\n\t indxImg=",indxImg,"\t image shape=",segmentedIm.shape

        X_train[indxImg] = cv2.resize(originalIm, (128, 128)) #originalIm
        y_train[indxImg] = cv2.resize(segmentedIm, (128, 128))

        '''
        for indxAngle,angle in enumerate(angles):
            for indxShift,shift in enumerate(shifts):
    
                M = cv2.getRotationMatrix2D((128/2,128/2),angle,1)
                shiftM = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
                rotatedIm = cv2.warpAffine(originalIm,M,(128,128))
                rotatedSegmentedIm = cv2.warpAffine(segmentedIm,M,(128,128))
                rotatedShiftedIm = cv2.warpAffine(rotatedIm,shiftM,(128,128))
                rotatedSegmentedShiftedIm = cv2.warpAffine(rotatedSegmentedIm,shiftM,(128,128))
                X_train[total]=rotatedShiftedIm
                y_train[total]=rotatedSegmentedShiftedIm
    
                cv2.imwrite(cwd+"//newData//"+str(indxImg)+"_"+str(indxAngle)+"_"+str(indxShift)+"_shift.jpg",rotatedShiftedIm)
                cv2.imwrite(cwd+"//newData//"+str(indxImg)+"_"+str(indxAngle)+"_"+str(indxShift)+"_segment.jpg",rotatedSegmentedShiftedIm)
    
    
                total+=1
    
        '''

        # showImage("train",originalIm)
        # showImage("test",segmentedIm)

    except Exception as e:
        noException+=1
        print "\n\t e=",e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no", exc_tb.tb_lineno)
        #input("check exception")


print "\n\t noException=",noException
tests = os.listdir(cwd+'/newData/test/')#["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
noTestImages=len(tests)

print "\n\t noTestImages=",noTestImages


X_test = np.zeros((noTestImages,128,128,3))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
testException=0

for pos in range(len(tests)):
    try:

        temp=cv2.imread(cwd+'/newData/test/'+tests[pos])
        #print "\n\t test size",temp.shape
        #showImage(str(pos),temp)

        im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        ret2, th2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if blackOnWhite == 1:
            temp = (255 - temp)

        X_test[pos] = cv2.resize(temp,(128,128))
        X_test1.append(temp)
    except Exception as e:
        print "\n\t file name =",tests[pos]
        testException+=1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no in test images=", exc_tb.tb_lineno)


print "\n\t testException=",testException

X_train-=128.0
X_train/=128.0
y_train-=128.0
y_train/=128.0
X_test-=128.0
X_test/=128.0

print "1.X_train shape=",X_train.shape
print "2.y_train shape=",X_train.shape
print "3.X_test shape=",X_test.shape

#
# meen = np.mean(X_train,axis=(0,1,2))
# std = np.std(X_train,axis=(0,1,2))
# X_train-=meen
# X_train/=std
#
# #y_train-=meen
# y_train/=255
#


def createModel():
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    clf = Sequential()

    clf.add(Convolution2D(filters=64,kernel_size=(5,3),input_shape=(128,128,3), padding='same'))
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

    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))



    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    #clf.add(MaxPooling2D(pool_size=(2,2),, strides=(1,1))

    clf.add(Convolution2D(filters=512*2,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024*2,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))


    clf.add(Convolution2D(filters=512*2,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))


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

    clf.compile(optimizer=adam,loss='mse',metrics=['mae'])
    #clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return clf

def createModelOriginal():
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    clf = Sequential()

    clf.add(Convolution2D(filters=64,kernel_size=(3,3),input_shape=(128,128,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2))) # 1

    clf.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))# 32 2

    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(1,1))) # 3

    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2))) # 4

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(UpSampling2D((2,2)))
    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
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

    clf.compile(optimizer=adam,loss='mse',metrics=['mae'])

    return clf

def createModel1():

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    clf = Sequential()

    clf.add(Convolution2D(filters=64,kernel_size=(3,3),input_shape=(128,128,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))
    #clf.add()

    '''
    clf.add(Convolution2D(filters=128,kernel_size=(7,3),padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))

    clf.add(Convolution2D(filters=256,kernel_size=(7,5), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(1,1)))

    clf.add(Convolution2D(filters=256,kernel_size=(10,10), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2,2)))

    clf.add(Convolution2D(filters=512,kernel_size=(10,5), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=1024,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=2048,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))

    clf.add(UpSampling2D((2,2)))
    clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
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
    '''
    clf.add(Convolution2D(3, (3, 3), padding='same'))
    clf.add(Activation('tanh'))

    #clf.compile(optimizer=adam,loss='mse',metrics=['mae'])
    clf.compile(optimizer=adam,loss='mse',metrics=['mae'])

    return clf


#base CV structure
def get_callbacks(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    msave =ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1)

    return [es, msave]


file_path = cwd+"//models//model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=10)

clf=createModel()
#clf=createModelOriginal()

model_json=clf.to_json()
with open(cwd+"modelArch.json", "w") as json_file:
    json_file.write(model_json)

print clf.summary()

#keras.callbacks.ModelCheckpoint(cwd+'//models//', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

clf.fit(X_train,y_train,batch_size=30, epochs=200,validation_split=0.2,callbacks=callbacks,shuffle=True,verbose=2)
#clf.save(cwd+'//models//model-10.h5')

sys.stdout.flush()
y_out = clf.predict(X_test)
y_out*=128.0
y_out+=128.0



for y in range(y_out.shape[0]):
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (h, w)) #originalIm
    cv2.imwrite(cwd+"//results//"+'y'+str(y)+'t.jpg',X_test1[y])
    cv2.imwrite(cwd+"//results//"+'y'+str(y)+'s1gray.jpg',tmp)

