# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:33:27 2019

@author: 方嘉祥
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import threading

answer = []
picture_num = 0
print('-----------------------------------------------------')
print('please restart your kernel after closing this program')
print('-----------------------------------------------------')
class MainWindow (QMainWindow):
    def __init__ (self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('./F74056166.ui', self)
        self.Connect_btn()
        
    def Connect_btn(self):
        self.Button_1_1.clicked.connect(self.Button_1_1_function)
        self.Button_2_1.clicked.connect(self.Button_2_1_function)
        self.Button_3_1.clicked.connect(self.Button_3_1_function)
        self.Button_3_2.clicked.connect(self.Button_3_2_function)
        self.Button_4_1.clicked.connect(self.Button_4_1_function)
        
    def Button_1_1_function(self):
        imgL = cv2.imread('imL.png',0)
        imgR = cv2.imread('imR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        disparity = stereo.compute(imgL,imgR)
        #cv2.imshow("disparity", disparity)
        plt.imshow(disparity,'gray')
        plt.show()
    def Button_2_1_function(self):
        cap = cv2.VideoCapture('bgSub.mp4')
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        
        while(1):
            ret, frame = cap.read()
        
            fgmask = fgbg.apply(frame)
        
            cv2.imshow('base frame',frame)
            cv2.imshow('frame',fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 50:
                break
        cap.release()
        #cv2.destroyAllWindows()
    def Button_3_1_function(self):
        cap = cv2.VideoCapture('featureTracking.mp4')
        
        ret, frame = cap.read()
        frame = cv2.convertScaleAbs(frame)
        
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 30.0
        params.maxArea = 100.0
        params.filterByCircularity = True
        params.minCircularity = 0.83
        params.filterByArea = True
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)
        for each_key in keypoints:
            keyx, keyy = each_key.pt
            keyx, keyy = int(keyx), int(keyy)
            cv2.rectangle(frame, (keyx-5, keyy-5), (keyx+5, keyy+5), (0, 0, 255), 2)
            print([keyx, keyy])
        #print(keypoints[0])
        #im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), flags = 0)
        cv2.imshow("Keypoints", frame)
        cap.release()
        print('3.1')
    def Button_3_2_function(self):
        cap = cv2.VideoCapture('featureTracking.mp4')
        
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 30.0
        params.maxArea = 100.0
        params.filterByCircularity = True
        params.minCircularity = 0.83
        params.filterByArea = True
        detector = cv2.SimpleBlobDetector_create(params)
        """
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        """
        lk_params = dict( winSize  = (21,21),
                  maxLevel = 3, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(old_frame)
        p0 = []
        for each_key in keypoints:
            keyx, keyy = each_key.pt
            #keyx, keyy = int(keyx), int(keyy)
            #keyx, keyy = float(keyx), float(keyy)
            p0.append([keyx, keyy])
        p0 = np.array(p0,dtype=np.float32).reshape(-1,1,2)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #print(p0)
        mask = np.zeros_like(old_frame)
        
        
        while(1):
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]
                    
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), (0, 0, 255), 2)
                frame = cv2.circle(frame,(a,b),5,(0, 0, 255),-1)
            img = cv2.add(frame,mask)
            
            cv2.imshow('frame',img)
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            
            k = cv2.waitKey(30) & 0xff

        cap.release()
        
    def Button_4_1_function(self):
        intrinsic = np.array([[2225.49585482, 0, 1025.5459589], [0, 2225.18414074, 1038.58518846], [0, 0, 1]],dtype=np.float32)
        distortion = np.array([-0.12874225, 0.0957782, -0.00099125, 0.00000278, 0.0022925], dtype=np.float32)
        
        rotation_bmp1 = np.array([[-0.97157425, -0.01827487, 0.23602862], [0.07148055, -0.97312723, 0.2188925], [0.22568565, 0.22954177, 0.94677165]], dtype=np.float32)
        rotation_bmp2 = np.array([[-0.8884799, -0.14530922, -0.435303],[0.07148066, -0.98078915, 0.18150248], [-0.45331444, 0.13014556, 0.88179825]], dtype=np.float32)
        rotation_bmp3 = np.array([[-0.52390938, 0.22312793, 0.82202974], [0.00530458, -0.96420621, 0.26510046], [0.85175749, 0.14324914, 0.50397308]], dtype=np.float32)
        rotation_bmp4 = np.array([[-0.63108673, 0.53013053, 0.566296], [0.13263301, -0.64553994, 0.75212145], [0.76428923, 0.54976341, 0.33707888]], dtype=np.float32)
        rotation_bmp5 = np.array([[-0.87676843, -0.23020567, 0.42223508], [0.19708207, -0.97286949, -0.12117596], [0.43867502, -0.02302829, 0.89835067]], dtype=np.float32)

        r1 = np.array(cv2.Rodrigues(rotation_bmp1))
        r2 = np.array(cv2.Rodrigues(rotation_bmp2))
        r3 = np.array(cv2.Rodrigues(rotation_bmp3))
        r4 = np.array(cv2.Rodrigues(rotation_bmp4))
        r5 = np.array(cv2.Rodrigues(rotation_bmp5))

        t1 = np.array([[6.81253889], [3.37330384], [16.71572319]], dtype=np.float32)
        t2 = np.array([[3.3925504], [4.36149229], [22.15957429]], dtype=np.float32)
        t3 = np.array([[2.68774801], [4.70990021], [12.98147662]], dtype=np.float32)
        t4 = np.array([[1.22781875], [3.48023006], [10.9840538]], dtype=np.float32)
        t5 = np.array([[4.43641198], [0.67177428], [16.24069227]], dtype=np.float32)
        
        images = [cv2.imread('%d.bmp' % (i), cv2.IMREAD_GRAYSCALE) for i in range(1, 6)]

        corner = np.array([(3,3,-4), (1,1,0), (1,5,0), (5,5,0), (5,1,0)], dtype=np.float32)
        
        
        
        def draw_picture(image, r, t):
            #r = tuple(r)                        
            """
            print(type(r))
            print(type(t))
            print(type(intrinsic))
            print(type(distortion))
            """
            print(type(image))
            """
            print('corner--------------------')
            print(corner)
            print('r--------------------')
            print(r)
            print('t--------------------')
            print(t)
            print('intrinsic--------------------')
            print(intrinsic)
            print('distortion--------------------')
            print(distortion)
            """
            corners, _ = cv2.projectPoints(corner, r, t, intrinsic, distortion)

            corners = np.squeeze(corners, axis=1)
            corners = [tuple(c) for c in corners]
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            image = cv2.line(image, corners[0], corners[1], [0, 0, 255], 10)
            image = cv2.line(image, corners[0], corners[2], [0, 0, 255], 10)
            image = cv2.line(image, corners[0], corners[3], [0, 0, 255], 10)
            image = cv2.line(image, corners[0], corners[4], [0, 0, 255], 10)
            
            image = cv2.line(image, corners[1], corners[2], [0, 0, 255], 10)
            image = cv2.line(image, corners[1], corners[4], [0, 0, 255], 10)
            
            image = cv2.line(image, corners[2], corners[3], [0, 0, 255], 10)
            
            image = cv2.line(image, corners[3], corners[4], [0, 0, 255], 10)
            
            image = cv2.resize(image, (512, 512))

            return image


        
        answer.clear()
        answer.append(draw_picture(images[0], r1[0], t1))
        answer.append(draw_picture(images[1], r2[0], t2))
        answer.append(draw_picture(images[2], r3[0], t3))
        answer.append(draw_picture(images[3], r4[0], t4))
        answer.append(draw_picture(images[4], r5[0], t5))

        

                
        cv2.imshow('frame',answer[picture_num])


        def changeImage():
            global picture_num
            print('Hello Timer!')
            picture_num = picture_num + 1
            if picture_num == len(answer):
                picture_num = 0
            cv2.imshow('frame',answer[picture_num])
            global timer
            timer = threading.Timer(0.5, changeImage)
            timer.start()
            
        timer = threading.Timer(0.5, changeImage)
        timer.start()
            
        print('4.1')
    """
    def changeImage(self):
        print('change')
        picture_num += 1
        if picture_num == len(answer):
            picture_num = 0
        cv2.imshow('frame',answer[picture_num])
        
"""
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    global timer
    timer.cancel()

