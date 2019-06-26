'''
    aim of this code is to separate train images and segment images
'''

import os
import cv2

#allDataPath="/home/kapitsa/Documents/Dataset/documentLayoutAnalysis/uw3-framed-lines-degraded-000//"
allDataPath="/home/kapitsa/pyCharm/segmentation/seg-master/data//"
path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//"

img=""
seg=""

allImgs=os.listdir(allDataPath)

imgList=[]
segList=[]

for indx,name in enumerate(allImgs):


    if indx%100==0:
        print("\n\t index=",indx)

    img = cv2.imread(allDataPath + name)

    # name1 = name.split(".png")[0]
    # name2 = name1 + ".lines.png"

    if "framed" in name and not "lines" in name:
        imgList.append(name)
        cv2.imwrite(path+"images2//"+name,img)
    elif "lines" in name:
        segList.append(name)
        cv2.imwrite(path+"segment2//"+name,img)

    img=None
print("\n\t text image count=",len(imgList))
print("\n\t segImage count=",len(segList))
print("All count=",len(allImgs))