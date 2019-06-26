#THIS CODE IS FOR 2ND PUBLICATION

# the idea is to hash the fully connected layers output
# by hashing I am trying to check similarity between 2 images

import sys
from keras.models import Sequential,model_from_json,Model
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras import regularizers
from sklearn.model_selection import train_test_split
import os
import numpy as np
import glob
import cv2
import shutil

cwd=os.getcwd()+"//"

'''
    this path shows prediction result store location
'''

folder1 = cwd+"paper2"+"//predResults//nonTables//"
folder2=cwd+"//paper2"+"//predResults//table//"

def  delAll(folder):

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

delAll(folder1)
delAll(folder2)

np.random.seed(100)

'''
    at this location training model is stored
'''
saveModel=cwd+"paper2"+"//models//"
blackOnWhite=1

'''
    ALL DATA PATHS FOR TRAINING DATA
'''
# tabPaths="/home/kapitsa/PycharmProjects/MyOCRService/myTable/tables//"
# #tablespath=glob.glob(cwd+"\\tables\\"+"/*.jpg")
# nonTable="/home/kapitsa/PycharmProjects/MyOCRService/myTable/nonTables//"

'''
    this data is used for training
'''
tabPaths=cwd+"//paper2//trainingData//tables//"
nonTable=cwd+"//paper2//trainingData//nonTables//"

'''
    COLLECT TABLES AND NON TABLES PATH
'''

tablespath=glob.glob(tabPaths+ "/*.jpg")+glob.glob(tabPaths+ "/*.png")+glob.glob(tabPaths+ "/*.bmp")
noTables=len(tablespath)

nonTablePath=glob.glob(nonTable+ "/*.bmp")+glob.glob(nonTable+ "/*.jpg")+glob.glob(nonTable+ "/*.png")
noNonTables=len(nonTablePath)

print("\n\t TOTAL NO OF TABLE TRAINING IMAGES=",noNonTables)
print("\n\t TOTAL NO OF NON TABLE IMAGES=",noTables)

'''
    train data contains full paths for training images
'''
trainData=tablespath+nonTablePath

'''
    training image size
'''
rows,columns,channels=32,32,3


#print(tablespath)

'''
    this function reads images containing table as well as non table,
    it resizes those images and returns them
'''

def readFile1(fileRead):

    readImages=[]
    #img=cv2.imread(location+fileRead)

    countExceptins=0
    for indx,f in enumerate(fileRead):
        try:
            img = cv2.imread(f)
            img=cv2.resize(img,(rows,columns),interpolation=cv2.INTER_CUBIC)
            readImages.append(img)
        except Exception as e:
            countExceptins+=1
            print("\n\t file=",f,"\n\t is file=",os.path.isfile(f),"\t indx=",indx)
            print("\n\t EXCEPTION WHILE READING FILE!")
            #print("\n\t shape=",img.shape)
            pass

    print("\n\t  countExceptins=",countExceptins)
    return readImages

# tabImages=readFile(tablespath)
# print("\n\t readImages=",len(tabImages))

# nonTabImages=readFile(nonTablePath)
# print("\n\t nonTabImages=",len(nonTabImages))

'''
    reads single image
'''
def readImage(filePath,indx):

    #readImages=[]
    #img=cv2.imread(location+fileRead)

    countExceptins=0

    try:
        img = cv2.imread(filePath)
        img=cv2.resize(img,(rows,columns),interpolation=cv2.INTER_CUBIC)
        #readImages.append(img)
    except Exception as e:
        countExceptins+=1
        print("\n\t file=",filePath,"\n\t is file=",os.path.isfile(filePath),"\t indx=",indx)
        print("\n\t exception in readImage")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t Line no:", exc_tb.tb_lineno)

        #print("\n\t shape=",img.shape)
        pass
        print("\n\t  countExceptins=",countExceptins)
    return img


'''
    prep_data requires list of full paths
'''
def prep_data1(images):

    try:

        count=len(images)

        data=np.ndarray((count,channels,rows,columns),dtype=np.uint8)

        data1={}

        image_file=images

        #for i,image_file in enumerate(images):

        image=readImage(image_file,-1)


        data=image.T
        data1=image.T

    except Exception as e:

        print "\n\t Exception in prep_data"
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t LINE NO OF EXCEPTION=", exc_tb.tb_lineno)
        print("\n\t Exception e=",e)



    return data,data1,image



'''
    prep_data requires list of full paths
'''
def prep_data(images):

    try:

        count=len(images)

        data=np.ndarray((count,channels,rows,columns),dtype=np.uint8)

        data1={}

        for i,image_file in enumerate(images):

            image=readImage(image_file,i)

            if image==None:
                continue

            data[i]=image.T
            data1[image_file]=image.T
            if i%1000 ==0:
                print "Processed {} of {}".format(i,count)
    except Exception as e:

        print("\n\n")
        print("-"*10)
        print "\n\t Exception in prep_data"
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t LINE NO OF EXCEPTION=", exc_tb.tb_lineno)
        print("\n\t Exception e=",e)
        print("##"*10)
        print("\n\n")

    return data,data1


train,test1=prep_data(trainData)

print("\n\t training data shape shape=",train.shape)
print("\n\t test1 shape=",test1.__sizeof__())

label=[]

'''
    BELOW PART CREATES LABELS FOR TRAINING DATA
    ASSUMING 1ST LABELS ARE TABLES AND REMAINING NON TABLES
'''
for i in range(noTables):
    label.append(1)

for i in range(noNonTables):
    label.append(0)

'''
print("\n\t label[0]=",label[0])
print("\n\t label[last]=",label[len(label)-1])
'''

'''
    reshape train
    change type to float
'''
train=train.reshape(-1,rows,columns,3)
x_train=train.astype('float32')
x_train/=255

'''
    change type of labels to cateogorical
    perform train test split
'''

y_train=np_utils.to_categorical(label)

print("\n\t y_train=",len(y_train))


#print("\n\t before x_train=",x_train.shape)
#print("\n\t before y_train=",y_train.shape)


x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=100,shuffle=True)

#print("\n\t y_train",y_train)

# for eleTrain in y_train:
#     print("\n\t eleTrain=",eleTrain)

#print ("\n\t SHAPE OF THE DATA FOR TRAIN/ TEST SPLIT::::")
#print "\n\t x_train shape=",x_train.shape,"\t x_test.shape=",x_test.shape,"\t y_train.shape=",y_train.shape,"\t ,y_test.shape=",y_test.shape


def build_bottleneck_model(model):
    #print("**")

    output=model.layers[15].output
    bottleneck_model=Model(model.input,output)
    return bottleneck_model


def DeepModel():

    model=Sequential()
    model.add(Conv2D(2*512,(3,3),input_shape=(rows,columns,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(1024,(1,1)))
    model.add(MaxPool2D(pool_size=(1,1)))
    model.add(Activation('relu'))


    model.add(Conv2D(64,(3,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64*2,(3,3)))
    model.add(MaxPool2D(pool_size=(1,1)))
    model.add(Activation('relu'))


    model.add(Conv2D(32,(3,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    filePath1=cwd+"paper2//"+"models// weight{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint=ModelCheckpoint(filePath1,monitor='val_acc',verbose=1,save_best_only=True)
    callbacks=[EarlyStopping(monitor='val_loss',patience=10,verbose=0),checkpoint]


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    #batch_size=8

    #model.fit(x_train,y_train,batch_size=128,nb_epoch=50,verbose=1,validation_data=(x_test,y_test),callbacks=callbacks)
    model.fit(x_train,y_train,batch_size=128,nb_epoch=50,verbose=1,validation_split=0.3,callbacks=callbacks)

    return model,filePath1

model,filePath1=DeepModel()

#print("\n\t model summary=",model.summary())
#print("\n\t filePath1=",filePath1)

model_json=model.to_json()

with open(saveModel+"model2.json","w") as json_file:
    json_file.write(model_json)

# serialize model 2 hdf5
model.save_weights(saveModel+"model2.h5")
print("\n\t SAVED MODEL WEIGHTS TO DISK!!!",os.path.isfile("model2.h5"))

json_file=open(saveModel+"model2.json","r")

loaded_model_json=json_file.read()
json_file.close()

loaded_model=model_from_json(loaded_model_json)

'''
    load weights into new model
'''

print("\n\t LOADED MODEL FROM DISK!!")
loaded_model.load_weights(saveModel+"model2.h5")
#loaded_model.compile('sgd','mse')

print("\n\t loaded_model summary=",loaded_model.summary())

bottleneck_model=build_bottleneck_model(loaded_model)
#print("\n\t bottleneck_model summary=",bottleneck_model.summary())


# /home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper2/model

def testModel():

    test_data=cwd+"//paper2//test_images//tables//"
    test_data_path=test_data
    #+glob.glob(dataPath + "/*.bmp")
    test_images =glob.glob(test_data+"/*.jpg")+glob.glob(test_data + "/*.bmp")+glob.glob(test_data + "/*.png")
    #test_images=[i for i in os.listdir(test_data)]
    print("\n\t no of test images=",len(test_images))
    ###############################

    try:

        for t in test_images:
            #test,test1,image =prep_data1(test_data_path+t)
            test,test1,image =prep_data1(t)
            writeTestImage=cv2.imread(t)

            print("\n\t TEST SHAPE:{}".format(test.shape))

            test = test.reshape(-1, rows, columns, 3)
            # print("\n\t TEST RESHAPE:{}".format(test.shape))
            x_test = test.astype("float32")
            x_test /= 255

            submission = model.predict(x_test, verbose=1)
            predVect=bottleneck_model.predict(x_test)

            print("\n\t predVect=",type(predVect))
            print("\n\t length=",predVect.shape)
            print("\n\t predVect=",predVect)


            for indx, val in enumerate(submission):

                print("\n\t probability of being non table ", val[0], "\t Table", val[1],"\t val=",val)

                # p=str(float(indx))+"_"+str(val[1])+".jpg"
                p1 = str(float(val[1])) + "_" + str(indx) + ".jpg" # indicates table probability
                p0 = str(float(val[0])) + "_" + str(indx) + ".jpg" # indicates non table probability
                imagePath = test_images[indx]
                # print("\n\t is prediction image present=",os.path.isfile(imagePath))
                imageResult = image #cv2.imread(imagePath)

                try:
                    # /home/kapitsa/PycharmProjects/MyOCRService/myTable/predResults
                    if val[1] > 0.8:#table
                        # cv2.imwrite(cwd+"//paper2/test_images/prediction/table//"+p,imageResult)
                        #cv2.imwrite(folder2 + p1, imageResult)
                        cv2.imwrite(folder2 + p1,writeTestImage)
                        #print("\n\t writing at-->1",os.path.isdir(folder2),"\t ",folder2)
                        #input("check")
                    else:# non table
                        # cv2.imwrite(cwd+"//paper2/test_images/prediction//noTables//"+p1,imageResult)
                        #cv2.imwrite(folder1 + p0, imageResult)
                        cv2.imwrite(folder1 + p0,writeTestImage)
                        #print("\n\t writing at-->0",os.path.isdir(folder1),"\t ",folder1)


                except Exception as e:
                    print("\n\t exception while writing results")
                    #print("\n\t p=", p)

    # except Exception as e:
    #     print("**")

    # try:
    #
    #     #for t in test_images:
    #
    #     #print("\n\t is test file present=",os.path.isfile(test_data_path+t))
    #     test,test1 =prep_data(test_images)
    #     print("\n\t TEST SHAPE:{}".format(test.shape))
    #
    #     test=test.reshape(-1,rows,columns,3)
    #     #print("\n\t TEST RESHAPE:{}".format(test.shape))
    #     x_test=test.astype("float32")
    #     x_test/=255
    #     #print("\n\t x_test RESHAPE:{}".format(x_test.shape))
    #     #submission=loaded_model.predict(x_test,verbose=1)
    #     submission = model.predict(x_test, verbose=1)
    #     for indx,val in enumerate(submission):
    #
    #         print("\n\t probability of being table ",val[0],"\t Non Table",val[1])
    #
    #         #p=str(float(indx))+"_"+str(val[1])+".jpg"
    #         p = str(float(val[1])) + "_" + str(indx) + ".jpg"
    #         p1 = str(float(val[0])) + "_" + str(indx) + ".jpg"
    #         imagePath=test_images[indx]
    #         #print("\n\t is prediction image present=",os.path.isfile(imagePath))
    #         imageResult=cv2.imread(imagePath)
    #
    #         try:
    #             #/home/kapitsa/PycharmProjects/MyOCRService/myTable/predResults
    #             if val[1]>0.95:
    #                 #cv2.imwrite(cwd+"//paper2/test_images/prediction/table//"+p,imageResult)
    #                 cv2.imwrite(folder2+p,imageResult)
    #             else:
    #                 #cv2.imwrite(cwd+"//paper2/test_images/prediction//noTables//"+p1,imageResult)
    #                 cv2.imwrite(folder1+p1,imageResult)
    #
    #
    #         except Exception as e:
    #             print("\n\t exception while writing results")
    #             print("\n\t p=",p)
    #


    except Exception as e:
        print "\n\t Exception in test"
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t LINE NO OF EXCEPTION=", exc_tb.tb_lineno)
        print("\n\t Exception e=",e)






testModel()
























