'''
    aim of the code is to read trained models embeddings.
    these embeddings are clustered using LSH
'''

import pandas as pd
import numpy as np
import os
from keras.models import Sequential,model_from_json,Model
import glob
import cv2
#from paper2_3 import prep_data1
from lshash import LSHash
import os,sys
cwd=os.getcwd()+"//"
saveModel=cwd+"paper2"+"//models//"
d=4096

# df=pd.read_csv(cwd + "//paper2//test_images//out.csv")
# print("\n\t df shape=",df.shape)

rows,columns,channels=4*32,4*32,3
folder1 = cwd+"paper2"+"//predResults//nonTables//"
folder2=cwd+"//paper2"+"//predResults//table//"
folder3=cwd+"//paper2"+"//predResults//other//"

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
delAll(folder3)

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



'''
    below part load old model
'''

def loadOld():
    print("\n\t loads old model")

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

    return loaded_model

loaded_model=loadOld()
#print("\n\t original model=",loaded_model.summary())

def build_bottleneck_model(model):
    #print("**")

    output=model.layers[25].output
    bottleneck_model=Model(model.input,output)
    return bottleneck_model


bottleneck_model=build_bottleneck_model(loaded_model)
print("\n\t bottleneck_model summary=",bottleneck_model.summary())


d=4096
k=11
L = 5  # number of tables

lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
queryImageName = "c11.png"#"0.jpg"# "c11.png"


def testModel(model,bottleneck_model):

    #test_data=cwd+"//paper2//test_images//tables//"
    test_data=cwd+"//paper2//test_images//table2//"
    test_data_path=test_data
    #+glob.glob(dataPath + "/*.bmp")
    test_images =glob.glob(test_data+"/*.jpg")+glob.glob(test_data + "/*.bmp")+glob.glob(test_data + "/*.png")
    #test_images=[i for i in os.listdir(test_data)]
    #print("\n\t no of test images=",len(test_images))
    ###############################

    df = pd.DataFrame(index=range(len(test_images)),columns=range(d+1))# this .csv is used in LSH for vector embedding
    imageDict={}

    try:
        query=[]
        for rowNo,t in enumerate(test_images):
            #test,test1,image =prep_data1(test_data_path+t)
            test,test1,image =prep_data1(t)
            writeTestImage=cv2.imread(t)

            #print("\n\t TEST SHAPE:{}".format(test.shape))

            test = test.reshape(-1, rows, columns, 3)
            # print("\n\t TEST RESHAPE:{}".format(test.shape))
            x_test = test.astype("float32")
            x_test /= 255

            submission = model.predict(x_test, verbose=1)
            predVect=bottleneck_model.predict(x_test)

            print("\n\t image name=",t.split("/")[-1])
            #print("\n\t type of predVect=",type(predVect))
            print("\n\t length=",predVect.shape)
            #level = map(list, predVect[0][1])
            #print("\n\t predVect=>",predVect[0][1])
            #print "\n\t level=",level
            #print("\n\t predVect=>>>",predVect)


            curVect=[]

            for fillColIndx,fillEle in enumerate(predVect[0]):
                df.iloc[rowNo,(fillColIndx+1)]=predVect[0][fillColIndx]
                curVect.append(predVect[0][fillColIndx])
                #df.iloc[rowNo,1]=predVect[0][1]

            #print "\n\t\t curVect=:::: ",curVect
            imageDict[str(curVect)]=t.split("/")[-1]

            lsh.index(curVect)

            if t.split("/")[-1]==queryImageName:
                query=curVect

            for indx, val in enumerate(submission):

                print("\n\t probability of being non table ", val[0], "\t Table", val[1],"\t val=",val)

                # p=str(float(indx))+"_"+str(val[1])+".jpg"
                p1 = t.split("/")[-1]+"_"+str(float(val[1])) + "_" +str(float(val[0])) + "_"+ str(rowNo) + ".jpg" # indicates table probability
                p0 = t.split("/")[-1]+"_"+str(float(val[0])) + "_" + str(float(val[1])) + "_"+str(rowNo) + ".jpg" # indicates non table probability



                imagePath = test_images[indx]
                # print("\n\t is prediction image present=",os.path.isfile(imagePath))
                imageResult = image #cv2.imread(imagePath)

                try:
                    # /home/kapitsa/PycharmProjects/MyOCRService/myTable/predResults
                    if val[1] > 0.7:#table
                        # cv2.imwrite(cwd+"//paper2/test_images/prediction/table//"+p,imageResult)
                        #cv2.imwrite(folder2 + p1, imageResult)
                        cv2.imwrite(folder2 + p1,writeTestImage)
                        #print("\n\t writing at-->1",os.path.isdir(folder2),"\t ",folder2)
                        #input("check")
                        df.iloc[rowNo, 0] = t.split("/")[-1]
                    elif val[0] > 0.6:# non table
                        # cv2.imwrite(cwd+"//paper2/test_images/prediction//noTables//"+p1,imageResult)
                        #cv2.imwrite(folder1 + p0, imageResult)
                        cv2.imwrite(folder1 + p0,writeTestImage)
                        #print("\n\t writing at-->0",os.path.isdir(folder1),"\t ",folder1)
                        df.iloc[rowNo, 0] = t.split("/")[-1]

                    else:
                        cv2.imwrite(folder3 + p0,writeTestImage)
                        #print("\n\t writing at-->0",os.path.isdir(folder1),"\t ",folder1)
                        df.iloc[rowNo, 0] = t.split("/")[-1]


                except Exception as e:
                    print("\n\t exception while writing results")
                            #print("\n\t p=", p)

            df.to_csv(cwd+"//paper2//test_images//out.csv")
    except Exception as e:
        print "\n\t Exception in test"
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t LINE NO OF EXCEPTION=>>", exc_tb.tb_lineno)
        print("\n\t Exception e=",e)

    print "\n\t query length=",len(query)
    response = lsh.query(query, num_results=10, distance_func='euclidean')

    #response = lsh.query(curVect, num_results=3, distance_func='euclidean')

    print "\n\t query Image Name=",queryImageName
    print "\n\t query",query
    #print "\n\n\t response=",response,"\n"

    #print "\n\t dictionary=",imageDict

    '''
    for key in imageDict.keys():
        print "\n\t\t key=",key
        print "\n\t\t value=",imageDict[key]
    '''
    #print "\n\t query",query

    print("\n\t SIMILAR VECTORS TO QUERY--->")
    for indx, ele in enumerate(response):

        print "\n\t indx=", indx #, "\n\t\t ele=", ele[0]
        print "\n\t\t distance=",ele[1]
        #print "\n\t ele all=",ele
        level=[]
        for e in ele[0]:
            #level = map(list, ele[0])
            level.append(e)
        #print "\n\t level=",level

        if imageDict.__contains__(str(level)):
            print "\n\t image name=",imageDict[str(level)]

        else:
            print "\n\t key not present!!!"

testModel(loaded_model,bottleneck_model)

