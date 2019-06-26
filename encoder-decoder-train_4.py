'''

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
oldFiles=os.listdir(cwd+"results2//")

for old in oldFiles:
    try:

        os.remove("/home/kapitsa/PycharmProjects/segmentation//Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/results/"+old)
    except Exception as e:
        print "\n\t cant delete=",old
        pass


'''
this code is modified for new segmentaion
'''
'''
def showImage(name,image):
    print "\n\t image=",image.shape
    cv2.imshow(name,image)
    cv2.waitKey()
'''

def showImage(name,image):
    #cv2.imshow(name,image)
    plt.imshow(image)
    plt.show()
    #cv2.waitKey()

# path_x = cwd+'/newData/imageText1/' #only hands
# path_y = cwd+'/newData/segmentText1/' #segmented data

path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData//"
path_x = path+'/imageWord/' #only hands
path_y = path+'/segWord/' #segmented data

total = 0

dump=os.listdir(path_x)
dumpLen=len(dump)
print("\n\t dumpLen1=",dumpLen)

dump=os.listdir(path_x)
dumpLen= len(dump)
print("\n\t dumpLen2=",dumpLen)

maxImageProcess= dumpLen
#for pos in range(len(path_x)):

noException=0
blackOnWhite=0
bothBlackAndWhite=0 # considers black and white images by inverting

X_train=np.zeros((maxImageProcess,128,128,3))
y_train=np.zeros((maxImageProcess,128,128,3))

if bothBlackAndWhite==1:
    X_train=np.zeros((2**bothBlackAndWhite*maxImageProcess,128,128,3))
    y_train=np.zeros((2**bothBlackAndWhite*maxImageProcess,128,128,3))

for indxImg,img in enumerate(sorted(dump)):

    #print("\n\t img=",img,"\t ",os.path.isfile(path_x+img),"\t ",os.path.isdir(path_x))


    if indxImg %100==0:
        print "\n\tindxImg=",indxImg,"\t dumpLen=",dumpLen
    if indxImg>=maxImageProcess:
            break
    try:
        originalIm = cv2.imread(path_x+img)
        print "\n\t indxImg=",indxImg,"\t image shape=",originalIm.shape

        #img=img.split(".png")[0]
        #img=img+".lines.png"

        segmentedIm = cv2.imread(path_y+img)
        print("\n\t isFile=",os.path.isfile(path_y+img))
        print "\n\t 1indxImg=",indxImg,"\t image shape=",segmentedIm.shape
        print "\n\t 2indxImg=",indxImg,"\t image shape=",originalIm.shape
        #input("image read")
        X_train[indxImg] = cv2.resize(originalIm, (128, 128)) #originalIm
        y_train[indxImg] = cv2.resize(segmentedIm, (128, 128))


        if 0:#bothBlackAndWhite==1:
            X_train[(2**bothBlackAndWhite*indxImg)+1] = cv2.resize(255-originalIm, (128, 128))  # originalIm
            y_train[(2**bothBlackAndWhite*indxImg)+1] = cv2.resize(255-segmentedIm, (128, 128))

        '''
        showImage("train",originalIm)
        showImage("test",segmentedIm)
        '''
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

        # if 1:#bothBlackAndWhite==1:
        #     X_test1.append(255-temp)
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
    # comment down

    '''
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
    '''
    ### commnted up
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


def get_callbacks(filepath, patience=100):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    msave =ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1)

    return [es, msave]


file_path = cwd+"//models//model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=100)

clf=createModel()

model_json=clf.to_json()
with open(cwd+"modelArch.json", "w") as json_file:
    json_file.write(model_json)

#print clf.summary()

#keras.callbacks.ModelCheckpoint(cwd+'//models//', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

clf.fit(X_train,y_train,batch_size=30, epochs=30,validation_split=0.2,callbacks=callbacks,shuffle=True,verbose=2)
#clf.save(cwd+'//models//model-10.h5')
clf.save(file_path)

sys.stdout.flush()
y_out = clf.predict(X_test)
y_out*=128.0
y_out+=128.0


for y in range(y_out.shape[0]):
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (w,h)) #originalIm
    cv2.imwrite(cwd+"//results2//"+'y'+str(y)+'t.jpg',X_test1[y])
    cv2.imwrite(cwd+"//results2//"+'y'+str(y)+'s1gray.jpg',tmp)

