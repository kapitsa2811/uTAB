from slidingWindow_1 import *

'''
    this function calls sliding window
    
    location: it is assumed that at this location cropText,cropSegment (to store output of sliding_Window) are present
    it is also assumed that imageWord , segWord containing training images are present
    
'''

location = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData/"

outLocation=location #="/home/kapitsa/PycharmProjects/MyOCRService/images//slidingWindow//"
# 0.outLocation="/home/kapitsa/PycharmProjects/MyOCRService/images//slidingWindow//"
#0.testDataPath="/home/kapitsa/PycharmProjects/MyOCRService/images/slidingWindow//testData//" # all images in this folder are considered as a test images



'''
    below function creates crop of text and segment
'''
sliding_Window(location,outLocation)

'''
    prediction
'''
model=""

testDataPath="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/newData/test//"
#prediction(model,testDataPath,testDataPath)