'''

aim of code is to
'''


import numpy as np
np.random.seed(1000) # for reproducibility
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution2D
from keras.layers import Activation,SpatialDropout2D,ZeroPadding2D,Input, merge
from keras.layers import MaxPooling2D,UpSampling2D,AveragePooling2D
from keras.layers import Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import *
from keras.models import load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.backend import shape as tfShape
from keras.backend import int_shape as tfShape2
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

maxImageProcess=dumpLen
#for pos in range(len(path_x)):

noException=0
blackOnWhite=0
bothBlackAndWhite=0 # considers black and white images by inverting

X_train=np.zeros((maxImageProcess,384,384,3))
y_train=np.zeros((maxImageProcess,384,384,3))

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
        # print("\n\t isFile=",os.path.isfile(path_y+img))
        # print "\n\t 1indxImg=",indxImg,"\t image shape=",segmentedIm.shape
        # print "\n\t 2indxImg=",indxImg,"\t image shape=",originalIm.shape
        #input("image read")
        X_train[indxImg] = cv2.resize(originalIm, (384, 384)) #originalIm
        y_train[indxImg] = cv2.resize(segmentedIm, (384, 384))

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


X_test = np.zeros((noTestImages,384,384,3))
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

        X_test[pos] = cv2.resize(temp,(384,384))
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

'''
    original model
'''

def softmax4(x):
    """Custom softmax activation function for a 4D input tensor
    softmax along axis = 1
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    elif ndim == 4:
        e = K.exp(x - K.max(x, axis=1, keepdims=True))
        s = K.sum(e, axis=1, keepdims=True)
        return e / s
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))


def create_single_res_model1() :
    '''Create a PPT text detector model with single resolution support
    '''
    model = Sequential()
    # block 1

    model.add( ZeroPadding2D( padding = ( 3, 3 ), input_shape = ( 384,384,3 ) ) )
    #model.add( ZeroPadding2D( padding = ( 3, 3 ), input_shape = ( 3, None, None ) ) )
    model.add( Convolution2D( 16, 7, 7, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 16, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    
    # block 2
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 3
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 4
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 8, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 0, 1 ) ) )
    model.add( Convolution2D( 16 * 16, 1, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 3, 1, 1, border_mode='valid' ) )
    model.add( Activation( 'relu' ) )
    # block 5
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 12, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 3, 3 ) ) )
    model.add( Convolution2D( 8, 7, 7, border_mode='valid' ) )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 5, 5 ) ) )
    model.add( Convolution2D( 3, 11, 11, border_mode='valid' ) )

    #model.add( Activation( softmax4 ) )
    return model

def create_single_res_model() :
    '''Create a PPT text detector model with single resolution support
    '''
    model = Sequential()
    # block 1
    model.add( ZeroPadding2D( padding = ( 3, 3 ), input_shape = ( None, None,3 ) ) )
    model.add( Convolution2D( 16, 7, 7, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 16, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 2
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 3
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 4
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 8, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 0, 1 ) ) )
    model.add( Convolution2D( 16 * 16, 1, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 3, 1, 1, border_mode='valid' ) )
    model.add( Activation( 'relu' ) )
    # block 5
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 12, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 3, 3 ) ) )
    model.add( Convolution2D( 8, 7, 7, border_mode='valid' ) )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 5, 5 ) ) )
    model.add( Convolution2D( 3, 11, 11, border_mode='valid' ) )
    model.add( Activation( softmax4 ) )
    return model



def createModel():
    '''Create a PPT text detector model using multi-resolution responses
        '''
    base_model = create_single_res_model()
    # create a multi-resolution model
    #inputs = Input(shape=(3, None, None))

    inputs = Input(shape=(384,384,3))
    '''
    a2 = AveragePooling2D((2, 2))(inputs)
    a3 = AveragePooling2D((3, 3))(inputs)
    a4 = AveragePooling2D((4, 4))(inputs)
    '''
    # decode at each resolution
    p1 = base_model(inputs)
    '''
    p2 = base_model(a2)
    
    
    #p2 = UpSampling2D((2, 2))(p2)
    #print("\n\t p1=",tfShape2(p1),"\t p2=",tfShape2(p2),"\t type=",type(p1))
    p3 = base_model(a3)
    p4 = base_model(a4)
    # dropout
    d1 = SpatialDropout2D(0.25)(p1)

    d2 = SpatialDropout2D(0.25)(p2)

    #print("\n\t d1=",tfShape2(d1),"\t d2=",tfShape2(d2))

    d3 = SpatialDropout2D(0.25)(p3)
    d4 = SpatialDropout2D(0.25)(p4)
    # map to original resolution

    o2 = UpSampling2D((2, 2))(d2)
    o3 = UpSampling2D((3, 3))(d3)
    o4 = UpSampling2D((4, 4))(d4)

    # merge all response
    #f = merge([d1, o2, o4], mode='concat', concat_axis=1)
    f = merge([d1, o2, o3, o4], mode='concat', concat_axis=1,output_shape=(384,384,3))
    '''
    #f = merge([d1, d2], mode='concat', concat_axis=1)
    '''
    f_pad = ZeroPadding2D((5, 5))(f)

    
    print("\n\t 0.f =",tfShape2(f))
    print("\n\t 0.d1 =",tfShape2(d1))
    print("\n\t 0.o2 =",tfShape2(o2))
    print("\n\t 0.o3 =",tfShape2(o3))
    print("\n\t 0.o4 =",tfShape2(o4))

    #f_pad = ZeroPadding2D((5, 5))(f)

    #print("\n\t 1.f_pad =",tfShape2(f_pad))
    d1 = ZeroPadding2D((5, 5))(d1)
    f_pad =d1

    bottle = Convolution2D(8, 11, 11, activation='relu', name='bottle')(f_pad)
    output = Convolution2D(3, 1, 1, activation=softmax4)(bottle)

    #base_model.compile(optimizer='adam',loss='mse',metrics=['mae'])

    #print("\n\t inputs shape=",inputs.shape,"\t output=",tfShape2(output))
    #inputs = merge([inputs, inputs, inputs, inputs], mode='concat', concat_axis=1)

    #print("\n\t inputs shape=",inputs.shape,"\t type=",type(inputs))

    #inputs=cv2.resize(inputs,(1536, 384))
    model = Model(input=inputs, output=output)

    #print("\n\t inputs shape=",inputs.shape)
    #model = Model(input=inputs, output=f_pad)
    '''
    return base_model



def get_callbacks(filepath, patience=30):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #msave = ModelCheckpoint(filepath, save_best_only=True)
    msave =ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1)

    return [es, msave]


file_path = cwd+"//models//model_weights_newArch.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=100)

clf=createModel()
clf.compile(optimizer='adam',loss='mse',metrics=['mae'])

model_json=clf.to_json()
with open(cwd+"modelArch_newArch.json", "w") as json_file:
    json_file.write(model_json)

#print clf.summary()

#keras.callbacks.ModelCheckpoint(cwd+'//models//', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

clf.fit(X_train,y_train,batch_size=30, epochs=10,validation_split=0.2,callbacks=callbacks,shuffle=True,verbose=2)
#clf.save(cwd+'//models//model-10.h5')
clf.save(file_path)

sys.stdout.flush()
print("\n\t predict=")
y_out = clf.predict(X_test)


y_out*=128.0
y_out+=128.0

print("\n\t X_test shape=",X_test.shape,"\t y_out =",y_out.shape)

for y in range(y_out.shape[0]):
    #print("\n\t y=",y)
    h,w=X_test1[y].shape[0],X_test1[y].shape[1]
    tmp= cv2.resize(y_out[y], (w,h)) #originalIm
    cv2.imwrite(cwd+"//result3//"+'y'+str(y)+'t.jpg',X_test1[y])
    cv2.imwrite(cwd+"//result3//"+'y'+str(y)+'s1gray.jpg',tmp)

