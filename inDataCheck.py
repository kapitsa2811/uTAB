'''
this code helps in movement of data from sourcePath1 to newData folder
'''


import os
import shutil
import sys
sys.path.append("/home/aniket/PycharmProjects/dataGenerationImage/")
#from imageDataCreation_3 import *


cwd=os.getcwd()
path_x = cwd+'/newData/X/' #only hands
path_y = cwd+'/newData/segment/'   #Y2 segmented data
xImg=os.listdir(path_x)
yImg=os.listdir(path_y)

print "\n\t x len=",len(xImg)
print "\n\t y len=",len(yImg)


for f in xImg:
    os.remove(path_x+f)

for f in yImg:
    os.remove(path_y+f)


#print "\n\t name match=",len(set(xImg)&set(yImg))

sourcePath1="/home/aniket/Documents/Dataset/crop/fusion1/segment//"
sourcePath2="/home/aniket/Documents/Dataset/crop/fusion1/"
source=os.listdir(sourcePath1)
destination=path_y
exceptionCount=0

for file in source:
    try:
        shutil.copy2(sourcePath1+file,path_y)
        os.remove(sourcePath1+file)
        #shutil.copy2(sourcePath2 + file, path_x)
    except Exception as e:
        exceptionCount+=1

print "\n\t exceptionCount=",exceptionCount
shutil.rmtree(sourcePath1,ignore_errors=True)
source2=os.listdir(sourcePath2)

for file in source2:
    try:
        shutil.copy2(sourcePath2+file,path_x)
        os.remove(sourcePath2+file)
    except Exception as e:
        exceptionCount+=1

shutil.rmtree(sourcePath2,ignore_errors=True)
print "\n\t exceptionCount=",exceptionCount

print "\n\t x len=",len(xImg)
print "\n\t y len=",len(yImg)

#len(xImg)==0 and
# if len(yImg)==0:
#     shutil.move("/home/aniket/Documents/Dataset/crop/fusion1/segment//",path_y)
#     #os.rmdir("/home/aniket/Documents/Dataset/crop/fusion1/segment//")
#     #shutil.move("/home/aniket/Documents/Dataset/crop/fusion1//",path_x)




