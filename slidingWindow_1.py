import cv2
import numpy as np
import math
import os
import sys
import shutil



'''
    1.purpose of these code is to crop part of text image and corresponding part in segment image
    2. To check if it is propoerly working or not again cropped images are merged.
    3. slidingWindow_1.py implements the same functionality for multiple images.
    4. this code is integrated in "encoder-decoder-train_5.py"
    
'''

noPieces=5
blackOverWhite=1

'''
    below is cropping image dimension
'''
h,w=500,500
totImgHeight,totImgWidth=int(2000/noPieces)*noPieces,int(2000/noPieces)*noPieces # this is not final image dimension but intermediate
                                   # by thid dimension image will be used for croping
                                   # division process brings image in multiple of noPieces

def loadImage(location):

    '''

    :param inFile:
    path: base path
    path_x = imageWord contains images
    path_y = segWord contains
    :return:
    '''
    path = location#"/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData//"
    path_x = path + '/imageWord/'  # only hands
    path_y = path + '/segWord/'  # segmented data

    total = 0

    dump = os.listdir(path_x)
    dumpLen = len(dump)
    #print("\n\t dumpLen1=", dumpLen)

    dump = os.listdir(path_y)
    dumpLen = len(dump)
    #print("\n\t dumpLen2=", dumpLen)

    maxImageProcess = dumpLen
    # for pos in range(len(path_x)):

    noException = 0
    blackOnWhite = 0
    bothBlackAndWhite = 0  # considers black and white images by inverting


    '''
        below stores intermediate images which are used for cropping
    '''
    X_train = np.zeros((maxImageProcess, totImgHeight, totImgWidth, 3))
    y_train = np.zeros((maxImageProcess, totImgHeight, totImgWidth, 3))

    for indxImg, img in enumerate(sorted(dump)):

        # print("\n\t img=",img,"\t ",os.path.isfile(path_x+img),"\t ",os.path.isdir(path_x))

        if indxImg % 100 == 0:
            print "\n\tindxImg=", indxImg, "\t dumpLen=", dumpLen
        if indxImg >= maxImageProcess:
            break
        try:
            originalIm = cv2.imread(path_x + img)
            #print "\n\t indxImg=", indxImg, "\t image shape=", originalIm.shape

            img = img.split(".png")[0]
            # img=img+".lines.png"

            segmentedIm = cv2.imread(path_y + img)

            '''
                this brings image dimension in multiple of noPieces
            '''
            hNew, wNew = int(math.ceil(originalIm.shape[1] / noPieces) * noPieces), int(math.ceil(originalIm.shape[0] / noPieces) * noPieces)
            originalIm = cv2.resize(originalIm, (hNew, wNew))
            segmentedIm = cv2.resize(segmentedIm, (hNew, wNew))
            '''
            print("\n\t isFile=", os.path.isfile(path_y + img))
            print "\n\t 1indxImg=", indxImg, "\t image shape=", segmentedIm.shape
            print "\n\t 2indxImg=", indxImg, "\t image shape=", originalIm.shape
            '''
            # input("image read")
            X_train[indxImg] = cv2.resize(originalIm, (totImgHeight, totImgWidth))  # originalIm
            y_train[indxImg] = cv2.resize(segmentedIm, (totImgHeight, totImgWidth))


            '''
            showImage("train",originalIm)
            showImage("test",segmentedIm)
            '''
        except Exception as e:
            noException += 1
            print "\n\t e=", e
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("\n\t line no", exc_tb.tb_lineno)
            # input("check exception")

    print "\n\t noException=", noException

    '''
    img=cv2.imread(inFile)
    #print("\n\t read image size=",img.shape)
    '''

    return X_train,y_train

'''
    operates on single image and extracts crops from text image and segment image
'''

def croppaste(no,greyImage,greySegment,outLocation,trainData):

    '''

    :param no:
    :param greyImage: original input image
    :param greySegment: original input image segent
    :param outLocation:
    :return:
    '''

    h, w,c= greyImage.shape
    hS, wS,cS= greySegment.shape


    allSamples = []
    allSamplesSeg=[]
    samples_xy = []
    count=0

    '''
        sliding window width and height
    '''

    dh, dw = int(math.ceil(h / noPieces)), int(math.ceil(w / noPieces))
    hNew,wNew=dh*noPieces,dw*noPieces

    greyImage=cv2.resize(greyImage,(wNew,hNew))
    greySegment=cv2.resize(greyImage,(wNew,hNew))

    h, w,c= greyImage.shape
    hS, wS,cS= greySegment.shape

    #print("\n\t modified sizes=",h,w)


    heatMap = np.zeros((h, w,3), dtype=np.float32) # this is for storing

    #print("\n\t heatMap size=",heatMap.shape)


    if blackOverWhite==1:
        heatMap.fill(0)
    else:
        heatMap.fill(0)

    heatMapSeg = np.zeros((hS, wS,3), dtype=np.float32) # this is for storing
    tempText = np.zeros((hS, wS,3), dtype=np.float32) # this is for storing
    tempSeg = np.zeros((hS, wS,3), dtype=np.float32) # this is for storing
    tempText.fill(0)
    tempSeg.fill(0)
    heatMapSeg.fill(0)


    '''
    if not trainData==0:
        dh,dw=int(math.ceil(h/noPieces)),int(math.ceil(w/noPieces))
    elif trainData==0: # this is neural network input size
        dh,dw=128,128
    '''
    # print("\n\t h=",h,"\t w=",w)
    # print("\n\t dh=",dh,"\t dw=",dw)

    #shutil.rmtree('/home/me/test')

    '''
        below loop extracts all crop of original and segment image
    '''
    for y in range(0, h, dh):
        #print("\n\t y=",y,"\t range=",range(0, w - dw, 10))
        for x in range(0, w ,dw):
            #print(x)

            allSamples.append(greyImage[y:(y+dh), x:(x+dw)])

            if trainData==1:
                allSamplesSeg.append(greySegment[y:(y+dh), x:(x+dw)])

            '''
                extract original image
            '''
            img=greyImage[y:(y+dh), x:(x+dw)]

            tempText[y:(y + dh), x:(x + dw)] =img
            nm=str(no)+"_"+str(count)+"_"+str(y)+"_"+str(x)+".jpg"

            #print(nm)
            cv2.imwrite(outLocation+"//cropText//"+nm,img) # this is used for observing old inputput
            cv2.imwrite(outLocation+"//text//"+nm,img) # this is used for prediction

            samples_xy.append((x, y)) # this stores the sample co-rdinate

            '''
                extract segment
            '''

            if trainData==1:
                imgSeg=greySegment[y:(y+dh), x:(x+dw)]
                tempSeg[y:(y + dh), x:(x + dw)] =imgSeg
                nm=str(no)+"_"+str(count)+"_"+str(y)+"_"+str(x)+".jpg"
                cv2.imwrite(outLocation+"//cropSegment//"+nm,imgSeg)

            count=count+1


    #print("\n\t 1. All Samples=",len(allSamples),"\t count=",count)

    # cv2.imwrite("/home/kapitsa/PycharmProjects/MyOCRService/images/predictedSegment//"+"//tempSeg.jpg",tempSeg)
    # cv2.imwrite("/home/kapitsa/PycharmProjects/MyOCRService/images/predictedSegment//" + "//tempText.jpg", tempText)

    return samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw


'''
    this function take image and segment image and creates crop images of both and dump it
    
    1.creates 2 folders 1. text 2. segment at outLocation and stores output which are cropped images
    
    2. it assumes input present at location in 2 folders a) imageWord b)segWord
    
    allSamples: contains crop of text image
    allSamplesSeg: contains crop of segment image 
    
'''
def sliding_Window(location,outLocation):


    # location = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData/"
    # outLocation="/home/kapitsa/PycharmProjects/MyOCRService/images//slidingWindow//"

    #location="/home/kapitsa/PycharmProjects/MyOCRService/images//data//"


    # inFileImage=location+"A00ABIN.framed.lines.png" # original image
    # inFileSegment = location + "A00ABIN.framed.png" # segment image


    '''
        need to pass location of of image and from it extracts image and segment entire files (not crop)
        All files present at location are read

    '''
    X_train, y_train = loadImage(location)
    print("\n \t X_train len=",X_train.shape,"\t y_train =",y_train.shape)
    #greySegment = loadImage(inFileSegment)

    '''
        delete old data present
    '''
    if os.path.isdir(outLocation+"//cropText//"):
        shutil.rmtree(outLocation+"//cropText//")
        os.mkdir(outLocation+"//cropText//")
    else:
        os.mkdir(outLocation+"//cropText//")

    if os.path.isdir(outLocation + "//cropSegment//"):
        shutil.rmtree(outLocation+"//cropSegment//")
        os.mkdir(outLocation+"//cropSegment//")

    else:
        os.mkdir(outLocation+"//cropSegment//")


    if os.path.isdir(outLocation + "//predictedSegment//"):
        shutil.rmtree(outLocation+"//predictedSegment//")
        os.mkdir(outLocation+"//predictedSegment//")
    else:
        os.mkdir(outLocation+"//predictedSegment//")


    '''
        from the file names cropping of image and segment is done stores it at outLocation 
    '''

    for no in range(0,len(X_train)):

        #print("\n\t image no=",no)

        # if not no==1:
        #     continue
        #
        # if no>=1:
        #     break

        greyImage,greySegment=X_train[no],y_train[no] # entire images

        trainData = 1  # when training this variable is 1 otherwise 0 so segment image is empty

        '''
            operates on single image and extracts crops from text image ansd segment image
            samples_xy: it contains only start coordinates of the crop
            allSamples: all sample text images
            allSamplesSeg: all sample segment from text
            heatMap: blank image of similar size of greyImage
            heatMapSeg: blank image of similar size of
            dh,dw: height and width of crop greySegment   
        '''

        samples_xy,allSamples, allSamplesSeg, heatMap,heatMapSeg,dh, dw=croppaste(no,greyImage, greySegment, outLocation,trainData)



    return samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw

#samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw=sliding_Window()

'''
dummyPrediction: only to visualize how prediction will work
samples_xy: xy cropping coordinates only start to derive end cordinates add dw,dh 
allSamples: contains images
allSamplesSeg: contains segment images
dh: crop amount in y
dw: crop amount in x
'''
def dummyPrediction(samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw,outLocation):

    i = 0
    n = len(allSamples)
    location=outLocation

    print("\n\t allSamples=",len(allSamples))

    while i < n:
        subBatchSize = min(32, n - i)
        #print("\n\t subBatchSize =",subBatchSize,"\t i=",i)
        subBatch = allSamples[i:(i+subBatchSize)]

        networkInput = np.reshape(subBatch, (subBatchSize, dh, dw, 3))
        #print("\n\t networkInput size=",len(networkInput))

        '''
            operation on segment part are just to visualize output but its not going to 
            be part of experiment
        '''

        #networkOutput = model.predict(networkInput, subBatchSize, 0)

        networkOutput=allSamplesSeg[i:(i+subBatchSize)]

        for j in range(subBatchSize):
            globalSampleID = i + j
            x, y = samples_xy[globalSampleID]

            temp=networkOutput[j]
            #print("\n\t croped part shape",temp.shape)
            #print("\n\t y=",y,"\t (y+dh)=",(y+dh))

            #heatMap[(y):(y+dh), (x):(x+dw)].fill(networkOutput[j])

            heatMap[(y):(y + dh), (x):(x + dw),:]=networkOutput[j]

        i=i+1

    cv2.imwrite(location+str(i)+"_segment.jpg",heatMap)


#dummyPrediction(samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw)


'''
    1.take test image
    2. create crop from those image
    3. pass those crop to model for prediction
    4. take prediction from model
    5. merge those crop and create entire image
    6. save merged image
    model: trained model
'''

def prediction(model,testDataPath,outLocation):

    i = 0
    trainData=0 # indicate test image
    #testImage=""

    testImageList=os.listdir(testDataPath)

    '''
        delete old data present
    '''

    if os.path.isdir(outLocation+"//cropText//"):
        shutil.rmtree(outLocation+"//cropText//")
        os.mkdir(outLocation+"//cropText//")
    else:
        os.mkdir(outLocation+"//cropText//")

    if os.path.isdir(outLocation + "//cropSegment//"):
        shutil.rmtree(outLocation+"//cropSegment//")
        os.mkdir(outLocation+"//cropSegment//")
    else:
        os.mkdir(outLocation+"//cropSegment//")


    if os.path.isdir(outLocation+"//text//"):
        shutil.rmtree(outLocation+"//text//")
        os.mkdir(outLocation+"//text//")
    else:
        os.mkdir(outLocation+"//text//")

    for indx,name in enumerate(testImageList):

        # if not name.endswith(".jpg"):
        #     continue

        testImage=cv2.imread(testDataPath+name)
        #print("\n\t 0.size of test image=",testImage.shape)

        samples_xy, allSamples, allSamplesSeg, heatMap, heatMapSeg, dh, dw=croppaste(indx, testImage, testImage, outLocation, trainData)
        '''
        print("\n\t 1.allSamples=",len(allSamples))
        print("\n\t 2.allSamples=",allSamples[0].shape)
        print("\n\t 3.dh=", dh, "\t dw=", dw)
        print("\n\t 4.allSamples=",samples_xy)
        print("\n\t 5.len(samples_xy)=",len(samples_xy),"\n\\n")
        '''
        n=len(os.listdir(outLocation+"//text//")) # here test crop samples are present

        while i < (n):
            subBatchSize = min(30, n - i)
            #print("\n\t 6.subBatchSize =",subBatchSize,"\t i=",i,"\t n - i=",(n - i),"\t i=",i,"\t i:(i+subBatchSize)",(i+subBatchSize))
            subBatch = allSamples[i:(i+subBatchSize)]
            #print("\n\t subBatch len=",len(subBatch),"\t n=",n) #,"\t ",subBatch[0].shape,"\t subBatchSize =",subBatchSize,"\t type=",type(subBatch))
            subBatch2 =[]
            for allCrop in subBatch:
                #print("\n\t shape=",allCrop.shape)
                tempImg=cv2.resize(allCrop,(128,128))
                #print("\n\t tempImg shape=",tempImg.shape)
                subBatch2.append(tempImg)

            networkInput = np.array(subBatch2)
            #networkInput = np.reshape(subBatch2, (subBatchSize))
            #networkInput = np.reshape(subBatch, (subBatchSize,))
            #print("\n\t networkInput size=",len(networkInput))

            #operation on segment part are just to visualize output but its not going to
            #be part of experiment

            networkOutput = model.predict(networkInput)

            #networkOutput=allSamplesSeg[i:(i+subBatchSize)]
            #networkOutput=allSamples[i:(i+subBatchSize)]

            for j in range(subBatchSize):
                #print("\n\t j=",j)
                globalSampleID = i + j
                #globalSampleID = j
                x, y = samples_xy[globalSampleID]
                #print("\n\t 6.x=",x,"\t y=",y)
                temp=networkOutput[j]
                temp *= 128.0
                temp += 128.0

                #print("\n\t croped part shape",temp.shape)
                #print("\n\t y=",y,"\t (y+dh)=",(y+dh))

                #heatMap[(y):(y+dh), (x):(x+dw)].fill(networkOutput[j])
                #print("\n\t before insert shape=",temp.shape)
                #insertTemp=cv2.resize(temp,(dw,dh))
                insertTemp=cv2.resize(temp,(dw,dh))
                #print("\n\t after resize shape=",insertTemp.shape)
                #print("\n\t x=",x,"\t y=",y,"\t heatmap shape=",heatMap.shape)
                heatMap[(y):(y + dh), (x):(x + dw),:]=insertTemp #temp
                cv2.imwrite(outLocation + "//cropSegment//"+str(indx)+"_"+str(i)+"_"+str(j)+ "_segment.jpg", temp)
            #i=i+1
            i = i +subBatchSize


        '''
            input crops to model are deleted below
        '''
        if os.path.isdir(outLocation + "//text//"):
            shutil.rmtree(outLocation + "//text//")
            os.mkdir(outLocation + "//text//")
        else:
            os.mkdir(outLocation + "//text//")

        cv2.imwrite(outLocation+str(indx)+"_segment.jpg",heatMap)

#Prediction(samples_xy,allSamples,allSamplesSeg,heatMap,heatMapSeg,dh,dw)