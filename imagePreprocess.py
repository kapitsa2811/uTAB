'''
    this code performs image preprocessing operaions like erosion dilation
    In efforts to discover table using this code shown no progress.
'''

import cv2
import glob
import numpy as np
#from pythonRLSA import rlsa

path="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper//publicationData//"
#newPath = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8/test/blocks//
# "
newPath = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8/train/blocks//"
newPath = "/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8/test/blocks//"
#newPath1="/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8//results//"
basePath=path
expNo=str(8)
#imgPath=expNo+"//test/imageWord//"
#imgPath=expNo+"//train/imageWord//"
imgPath=expNo+"//test/imageWord//"
imageNames=glob.glob(path+imgPath+"*.jpg")
imageNames=imageNames+glob.glob(path+imgPath+"*.ppm")

print(len(imageNames))


threshold = (25, 20)
#wb_image_rlsa = rlsa_2d(wb_page_image, threshold)

def generate_blocks_dilation(img):
    kernel1 = np.ones((1, 3), np.uint8)
    kernel2 = np.ones((2, 1), np.uint8)

    '''
    ret, thresh1 = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV)
    '''
    i=10
    d_im=cv2.dilate(img,kernel1,iterations=i)
    #d_im=cv2.erode(d_im,kernel1,iterations=i)
    d_im1=cv2.dilate(img,kernel2,iterations=i)
    #d_im1=cv2.erode(d_im1,kernel2,iterations=i)
    d_im=d_im+d_im1
    #d_im=cv2.dilate(img,kernel,iterations=100)

    return d_im #cv2.dilate(thresh1, kernel, iterations=100)
    #image_rlsa_horizontal = rlsa.rlsa(threshold, True, False, 10)
    #return  image_rlsa_horizontal #rlsa_2d(img, threshold)



'''
for indx,imgNm in enumerate(imageNames):

    #print(imgNm)
    img=cv2.imread(imgNm)
    #img=img
    #img=255-img
    #print("\n\t img shape=",img.shape)
    #img=generate_blocks_dilation(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # function call


    image_rlsa_horizontal = rlsa.rlsa(image_binary, True,False,10)
    #image_rlsa_horizontal1 = rlsa.rlsa(image_binary, True,False,10)
    #image_rlsa_horizontal1 = rlsa.rlsa(image_binary, False,True,10)
    #image_dilate_erode = generate_blocks_dilation(img)

    nm=imgNm.split("/")[-1]
    nm=nm.split(".")[0]
    nm1=nm+"0.jpg"
    nm=nm+".jpg"
    #print("\n\t nm=",nm)


    #cv2.imwrite(newPath+nm1,image_rlsa_horizontal1)
    cv2.imwrite(newPath+nm,image_rlsa_horizontal)
    #cv2.imwrite(newPath+nm,image_dilate_erode)
    #image_rlsa_horizontal = rlsa.rlsa(image_binary, False,True, 10)

    #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8/train//blocks//v"+nm1,image_rlsa_horizontal)
'''


for indx,imgNm in enumerate(imageNames):

    #print(imgNm)
    img=cv2.imread(imgNm)
    #img=img
    #img=255-img
    #print("\n\t img shape=",img.shape)
    #img=generate_blocks_dilation(img)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # function call


    #image_rlsa_horizontal = rlsa.rlsa(image_binary, True,False,15)
    #image_rlsa_horizontal1 = rlsa.rlsa(image_binary, True,False,10)
    #image_rlsa_horizontal1 = rlsa.rlsa(image_binary, False,True,10)
    image_rlsa_horizontal = generate_blocks_dilation(img)

    nm=imgNm.split("/")[-1]
    nm=nm.split(".")[0]
    nm1=nm+".jpg"
    nm=nm+"5.jpg"
    #print("\n\t nm=",nm)


    cv2.imwrite(newPath+nm1,image_rlsa_horizontal)
    #cv2.imwrite(newPath+nm,image_rlsa_horizontal)
    #image_rlsa_horizontal = rlsa.rlsa(image_binary, False,True, 10)

    #cv2.imwrite("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper/publicationData/8/train//blocks//v"+nm1,image_rlsa_horizontal)
