import cv2
import numpy as np
import os


basePath="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/train/blocks//"
basePath2="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/8/train//"

path="/home/kapitsa/PycharmProjects/objectLocalization/DocumentTableSeg/DCEDN/publicationData/test//"


writepath=basePath2+"//contourBlocksTest//"

nm="0_slastNormalCanvas.jpg"



def block():

    for indx, name in enumerate(os.listdir(path)):

        try:
            #canny_img=cv2.imread(basePath+name)
            canny_img=cv2.imread(path+name)
            #canny_img=255-canny_img
            print("\n\t image size=",canny_img.shape)

            #im2, contours, hierarchy = cv2.findContours(canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            im_bw = cv2.cvtColor(canny_img, cv2.COLOR_RGB2GRAY)

            (thresh, im_bw) = cv2.threshold(im_bw,128, 255, 0)

            im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            try: hierarchy = hierarchy[0]
            except: hierarchy = []

            height, width, _ = canny_img.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            # computes the bounding box for the contour, and draws it on the frame,
            for contour, hier in zip(contours, hierarchy):

                (x,y,w,h) = cv2.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x+w, max_x)
                min_y, max_y = min(y, min_y), max(y+h, max_y)
                #if w > 80 and h > 80:
                cv2.rectangle(canny_img, (x,y), (x+w,y+h), (255, 0, 0), 2)
                #area1 = np.array([[250, 200], [300, 100], [450, 500], [100, 300]])
                #cv2.fillPoly(canny_img, pts=[np.array([[x,y],[x,y+h],[x+w,y],[x+w,y+h]])], color=(255, 255, 255))

                cv2.fillPoly(canny_img, pts=[np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])], color=(255, 255, 255))
                area2 = np.array([[1000, 200], [1500, 200], [1500, 400], [1000, 400]])

                #cv2.fillPoly(canny_img, [area2], (255, 255, 255))
            cv2.imwrite(writepath+name,canny_img)
        except Exception as e:
            print("\n\t exception image=",name)
            pass

block()
#if max_x - min_x > 0 and max_y - min_y > 0:
#cv2.rectangle(canny_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
#area1 = np.array([[250, 200], [300, 100], [450, 500], [100, 300]])
#cv2.fillPoly(np.array(canny_img), pts =[area1], color=(255,0,255))



'''
import matplotlib.pyplot as plt
img = np.zeros((1080, 1920, 3), np.uint8)
area1 = np.array([[250, 200], [300, 100], [750, 800], [100, 1000]])
area2 = np.array([[1000, 200], [1500, 200], [1500, 400], [1000, 400]])

cv2.fillPoly(img, [ area2], (255, 255, 255))

plt.imshow(img)
plt.show()
'''