
'''
    purpose of this code is to separate train and test data
'''
import cv2
import os

path="/home/kapitsa/pyCharm/segmentation/seg-master/data//"
writePath="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData//"

allImagesNames=os.listdir(path)
segLineImageName=[]
textImageName=[]

for imgName in allImagesNames:
    img=cv2.imread(path + imgName)
    if imgName.__contains__("lines"):
        segLineImageName.append(imgName)

        cv2.imwrite(writePath+"//segment//"+imgName,img)

    else:
        #print("\n\t name=",imgName)
        textImageName.append(imgName)
        cv2.imwrite(writePath+"//image//"+imgName,img)


print("\n\t no of segmented images=",len(segLineImageName))
print("\n\t no of text images=",len(textImageName))