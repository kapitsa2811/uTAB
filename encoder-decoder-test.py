import numpy as np
np.random.seed(1000) # for reproducibility
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D,UpSampling2D
from keras.layers import Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import *
from keras.models import load_model
from keras.models import model_from_json
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.backend import manual_variable_initialization
from slidingWindow_1 import *

manual_variable_initialization(True)
import os
import cv2
import sys

cwd=os.getcwd()+"//"

print os.path.isdir(cwd+"//results//")
print cwd+"//results//"

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
    #clf.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return clf


#clf=createModel()

print "\n\t file=",os.path.isfile(cwd+'modelArch.json')
model_json= open(cwd+'modelArch.json', 'r')
loaded_model = model_json.read()
model_json.close()
clf=model_from_json(loaded_model)
print "\n\t clf=",type(clf)
file_path = "/models/model_weights.hdf5"
print "\n\t weights=",os.path.isfile(cwd+file_path)

model=clf.load_weights(cwd+file_path)
#adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#clf.compile(optimizer=adam, loss='mse', metrics=['mae'])
clf.compile(optimizer='adam', loss='mse', metrics=['mae'])

#model.compile(optimizer=adam, loss='mse', metrics=['mae'])



testDataPath=cwd+'/newData//test//'
resultDataPath=cwd+"/newData/cropResults//"

# for indx,testImage in enumerate(X_test1):
#     cv2.imwrite(resultDataPath+'y'+str(y)+'t.jpg',X_test1[indx]) # this writes test image


'''
    only one time call is needed all test images are processed in one go
'''
prediction(clf,testDataPath,resultDataPath)


exit()

'''
tests = os.listdir(cwd+'/newData/test/')#["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
noTestImages=len(tests)
print "\n\t noTestImages=",noTestImages

X_test = np.zeros((noTestImages,128,128,3))
X_test1 =[] #np.zeros((noTestImages,512,512,3)) # original images
blackOnWhite=0
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


print "**"
y_out = clf.predict(X_test,verbose=0)
y_out*=128.0
y_out+=128.0


for y in range(y_out.shape[0]):
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (h, w)) #originalIm
    print "\n\t tmp=",tmp.shape
    cv2.imwrite(cwd+"//results//"+'y'+str(y)+'t.jpg',X_test1[y])
    cv2.imwrite(cwd+"//results//"+'y'+str(y)+'s1gray.jpg',tmp)

'''