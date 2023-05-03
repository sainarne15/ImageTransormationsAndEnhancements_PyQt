import math
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi
import sys
import cv2, imutils
from matplotlib import pyplot as plt
import numpy as np
from cmath import exp, pi
from skimage.metrics import structural_similarity as ssim


class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('transform.ui', self)

        self.image = None
        self.image1 =None #temp
        
        
        self.target_image = None
        self.tmp = None
        self.original_image = None
        self.tmptarget = None
        self.image_red = []
        self.image_green = []
        self.image_blue = []
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.openbutton.clicked.connect(self.open_img)
        self.openbutton2.clicked.connect(self.open_target_img)
        self.resetButton.clicked.connect(self.reset)
        self.clearButton.clicked.connect(self.clear)
        self.negativetransformation.toggled.connect(self.negative_transformation)
        self.fftimage.toggled.connect(self.fft_image) #fft_image
        self.ifftimage.toggled.connect(self.IFFT_image) #fft_image
        #self.fft2D.toggled.connect(self.FFT_2D) #fft_2Dimage
        #self.fft.toggled.connect(self.FFT) #fftimage
        self.bandpassfilter.toggled.connect(self.band_pass_filter)#band_pass_filter 
        self.unsharpmaskfilter.toggled.connect(self.unsharp_mask_filter)#unsharp_mask_filter 
        self.lowpassfilter.toggled.connect(self.Low_pass_filter) #Low Pass filter
        self.highpassfilter.toggled.connect(self.high_pass_filter) #High Pass Filter
        self.bilinearinterpolation.toggled.connect(self.bilinear_interpolation)  #bilinearinterpolation 
        self.bicubicinterpolation.toggled.connect(self.bicubic_interpolation)  #laplacianfilter
        self.laplacianfilter.toggled.connect(self.laplacian_filter)  #laplacianfilter
        self.meanfilter.toggled.connect(self.mean_filter) #meanFilter
        self.medianfilter.toggled.connect(self.median_filter) #medianFilter
        #self.bilinearinterpolation.toggled.connect(self.bilinear_interpolation)#bilinear_interpolation
        self.logtransformation.toggled.connect(self.log_transformation)
        self.powerlawtrans.toggled.connect(self.gamma_transformation)
        self.gammaParameter.valueChanged.connect(self.gamma_transformation)
        self.histequalization.toggled.connect(self.histogram_equalization)
        self.histshaping.toggled.connect(self.histogram_shaping)
        self.histmatching.toggled.connect(self.histogram_matching)
        self.transformButtonMatch.clicked.connect(self.trans_histogram_matching)
        self.transformButtonShape.clicked.connect(self.trans_histogram_shaping)
        self.transformButtonShape.setVisible(False)
        self.transformButtonMatch.setVisible(False)


    @pyqtSlot()
    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.tmp = self.image
        self.original_image = self.image
        self.display_image(2)

    def load_target_image(self, filename):
        self.image = cv2.resize(cv2.imread(filename), (200, 200), interpolation=cv2.INTER_CUBIC)
        self.tmptarget = self.image
        self.display_image(1)

    def reset(self):
        self.image = self.original_image
        self.display_image(2)
        self.ssimVal.setText(str(''))
        self.i_label_2.clear()

    def clear(self):
        self.t_label.clear()
        self.i_label.clear()
        self.i_label_2.clear()
        self.image = None
        self.target_image = None
        self.tmp = None
        self.ssimVal.setText(str(''))
        self.tmptarget = None

    def display_image(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()

        if window == 1:
            self.t_label.setPixmap(QPixmap.fromImage(img))
            self.t_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        if window == 2:
            self.i_label.setPixmap(QPixmap.fromImage(img))
            self.i_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            
        if window == 3:
            self.i_label_2.setPixmap(QPixmap.fromImage(img))
            self.i_label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    def display_image_error(self):
        qformat = QImage.Format_Indexed8

        if len(self.image1.shape) == 3:
            if(self.image1.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image1, self.image1.shape[1], self.image1.shape[0], self.image1.strides[0], qformat)
        img = img.rgbSwapped()

        self.i_label_2.setPixmap(QPixmap.fromImage(img))
        self.i_label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def display_image2(self):
        qformat = QImage.Format_Indexed8
        self.image = self.image.astype('uint8')
        self.image = imutils.resize(self.image, width=self.image.shape[1])
        print(self.image[1][1])

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat).rgbSwapped()
        p = image.scaled(460, 460, QtCore.Qt.KeepAspectRatio)
        self.i_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def open_img(self):
        filename, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users', "Image Files (*)")
        if filename:
            self.load_image(filename)
        else:
            print("Invalid Image")

    def open_target_img(self):
        filename1, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users', "Image Files (*)")
        if filename1:
            self.load_target_image(filename1)
        else:
            print("Invalid Image")

    def negative_transformation(self):
        self.reset()
        if self.image is not None:
            self.image = self.tmp
            self.image = np.max(self.image) - self.image
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)

    def log_transformation(self):
        self.reset()
        if self.image is not None:
            self.image = self.original_image

            c = 255 / np.log(1 + np.max(self.image))

            self.image = np.array(c * (np.log(self.image + 1)), dtype=np.uint8)
            
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)
            
            
            

    def gamma_transformation(self):
        self.reset()
        if self.image is not None:
            if self.powerlawtrans.isChecked():
                self.image = self.tmp
                gamma = self.gammaParameter.value()
                self.image = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
                self.display_image(2)
                self.t_label.setVisible(False)
                self.openbutton2.setVisible(False)
                self.transformButtonMatch.setVisible(False)
                self.transformButtonShape.setVisible(False)
                
                ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
                self.ssimVal.setText(str(ssim))
                self.display_image_error()
                self.t_label.setVisible(False)
                self.openbutton2.setVisible(False)
                self.transformButtonMatch.setVisible(False)

    def histogram(self, image):
        self.image = image
        img_list = []
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                for k in range(3):
                    img_list.append(self.image[i][j][k])

        list_len = len(img_list)
        image_red = []
        image_green = []
        image_blue = []

        for i in range(list_len):
            if i % 3 == 0:
                image_red.append(img_list[i])
            if i % 3 == 1:
                image_green.append(img_list[i])
            if i % 3 == 2:
                image_blue.append(img_list[i])

        hist_img_red = np.zeros((256,))
        hist_img_green = np.zeros((256,))
        hist_img_blue = np.zeros((256,))

        for i in range(256):
            hist_img_red[image_red[i]] += 1
            hist_img_green[image_green[i]] += 1
            hist_img_blue[image_blue[i]] += 1

        return hist_img_red, hist_img_green, hist_img_blue, image_red, image_green, image_blue

    def cumulative_histogram(self, hist_img_red, hist_img_green, hist_img_blue):
        e_hist_img_red = np.zeros((256,))
        e_hist_img_green = np.zeros((256,))
        e_hist_img_blue = np.zeros((256,))

        e_hist_img_red[0] = hist_img_red[0]
        e_hist_img_green[0] = hist_img_green[0]
        e_hist_img_blue[0] = hist_img_blue[0]

        for x in range(1, 256):
            e_hist_img_red[x] = e_hist_img_red[x - 1] + hist_img_red[x]
            e_hist_img_green[x] = e_hist_img_green[x - 1] + hist_img_green[x]
            e_hist_img_blue[x] = e_hist_img_blue[x - 1] + hist_img_blue[x]

        return e_hist_img_red, e_hist_img_green, e_hist_img_blue

    def histogram_equalization(self):
        self.reset()
        if self.image is not None:
            self.image = self.tmp

            img_list = []
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(3):
                        img_list.append(self.image[i][j][k])
            image_red = []
            image_green = []
            image_blue = []
            list_len = len(img_list)

            for i in range(list_len):
                if i % 3 == 0:
                    image_red.append(img_list[i])
                if i % 3 == 1:
                    image_green.append(img_list[i])
                if i % 3 == 2:
                    image_blue.append(img_list[i])

            hist_img_red = np.zeros((256,))
            hist_img_green = np.zeros((256,))
            hist_img_blue = np.zeros((256,))

            for i in range(256):
                hist_img_red[image_red[i]] += 1
                hist_img_green[image_green[i]] += 1
                hist_img_blue[image_blue[i]] += 1

            e_hist_img_red = np.zeros((256,))
            e_hist_img_green = np.zeros((256,))
            e_hist_img_blue = np.zeros((256,))

            e_hist_img_red[0] = hist_img_red[0]
            e_hist_img_green[0] = hist_img_green[0]
            e_hist_img_blue[0] = hist_img_blue[0]

            for x in range(1, 256):
                e_hist_img_red[x] = e_hist_img_red[x - 1] + hist_img_red[x]
                e_hist_img_green[x] = e_hist_img_green[x - 1] + hist_img_green[x]
                e_hist_img_blue[x] = e_hist_img_blue[x - 1] + hist_img_blue[x]

            final_hist_img_red = np.zeros((256,))
            final_hist_img_green = np.zeros((256,))
            final_hist_img_blue = np.zeros((256,))

            for x in range(256):
                final_hist_img_red[x] = 255 * (e_hist_img_red[x] - min(e_hist_img_red)) / (
                        max(e_hist_img_red) - min(e_hist_img_red))
                final_hist_img_green[x] = 255 * (e_hist_img_green[x] - min(e_hist_img_green)) / (
                        max(e_hist_img_green) - min(e_hist_img_green))
                final_hist_img_blue[x] = 255 * (e_hist_img_blue[x] - min(e_hist_img_blue)) / (
                        max(e_hist_img_blue) - min(e_hist_img_blue))

            image_red = np.array(image_red)
            image_green = np.array(image_green)
            image_blue = np.array(image_blue)

            final_hist_img_red = final_hist_img_red.astype('uint8')
            final_hist_img_green = final_hist_img_green.astype('uint8')
            final_hist_img_blue = final_hist_img_blue.astype('uint8')

            final_img_red = final_hist_img_red[image_red]
            final_img_green = final_hist_img_green[image_green]
            final_img_blue = final_hist_img_blue[image_blue]

            final_img_red = np.reshape(final_img_red, (self.image.shape[0], self.image.shape[1]))
            final_img_green = np.reshape(final_img_green, (self.image.shape[0], self.image.shape[1]))
            final_img_blue = np.reshape(final_img_blue, (self.image.shape[0], self.image.shape[1]))

            final_img = cv2.merge((final_img_blue, final_img_green, final_img_red))

            self.image = final_img
            self.display_image2()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)

    def histogram_matching(self):
        self.t_label.setVisible(True)
        self.openbutton2.setVisible(True)
        self.transformButtonMatch.setVisible(True)
        self.transformButtonShape.setVisible(False)

    def histogram_shaping(self):
        self.t_label.setVisible(True)
        self.openbutton2.setVisible(True)
        self.transformButtonMatch.setVisible(False)
        self.transformButtonShape.setVisible(True)

    def trans_histogram_shaping(self):
        self.reset()
        if self.image is not None:
            self.image = self.tmp
            self.tmptarget = cv2.resize(self.tmptarget, (self.image.shape[0],self.image.shape[1]), interpolation=cv2.INTER_CUBIC)
            final_img_red = np.zeros((self.image.shape[0], self.image.shape[1]))
            final_img_green = np.zeros((self.image.shape[0], self.image.shape[1]))
            final_img_blue = np.zeros((self.image.shape[0], self.image.shape[1]))

            hist_img_red, hist_img_green, hist_img_blue, image_red, image_green, image_blue = self.histogram(self.image)

            e_hist_img_red, e_hist_img_green, e_hist_img_blue = self.cumulative_histogram(hist_img_red, hist_img_green,
                                                                                    hist_img_blue)

            final_hist_img_red = np.zeros((256,))
            final_hist_img_green = np.zeros((256,))
            final_hist_img_blue = np.zeros((256,))

            for x in range(256):
                final_hist_img_red[x] = 255 * (e_hist_img_red[x] - min(e_hist_img_red)) / (
                            max(e_hist_img_red) - min(e_hist_img_red))
                final_hist_img_green[x] = 255 * (e_hist_img_green[x] - min(e_hist_img_green)) / (
                            max(e_hist_img_green) - min(e_hist_img_green))
                final_hist_img_blue[x] = 255 * (e_hist_img_blue[x] - min(e_hist_img_blue)) / (
                            max(e_hist_img_blue) - min(e_hist_img_blue))
            
            thist_img_red, thist_img_green, thist_img_blue, timage_red, timage_green, timage_blue = self.histogram(self.tmptarget)

            te_hist_img_red, te_hist_img_green, te_hist_img_blue = self.cumulative_histogram(thist_img_red, thist_img_green,
                                                                                       thist_img_blue)

            tfinal_hist_img_red = np.zeros((256,))
            tfinal_hist_img_green = np.zeros((256,))
            tfinal_hist_img_blue = np.zeros((256,))
            for x in range(256):
                tfinal_hist_img_red[x] = 255 * (te_hist_img_red[x] - min(te_hist_img_red)) / (
                            max(te_hist_img_red) - min(te_hist_img_red))
                tfinal_hist_img_green[x] = 255 * (te_hist_img_green[x] - min(te_hist_img_green)) / (
                            max(te_hist_img_green) - min(te_hist_img_green))
                tfinal_hist_img_blue[x] = 255 * (te_hist_img_blue[x] - min(te_hist_img_blue)) / (
                            max(te_hist_img_blue) - min(te_hist_img_blue))

            temp_red = np.zeros((256,))
            temp_green = np.zeros((256,))
            temp_blue = np.zeros((256,))

            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_red[j] >= final_hist_img_red[i]:
                        temp_red[i] = j
                        break
            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_green[j] >= final_hist_img_green[i]:
                        temp_green[i] = j
                        break
            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_blue[j] >= final_hist_img_blue[i]:
                        temp_blue[i] = j
                        break

            image_red = np.array(image_red)
            image_green = np.array(image_green)
            image_blue = np.array(image_blue)
            
           
            image_red = image_red.reshape(self.tmp.shape[0], self.tmp.shape[1])
            image_green = image_green.reshape(self.tmp.shape[0], self.tmp.shape[1])
            image_blue = image_blue.reshape(self.tmp.shape[0], self.tmp.shape[1])

            for i in range(final_img_red.shape[0]):
                for j in range(final_img_red.shape[1]):
                    final_img_red[i][j] = temp_red[image_red[i][j]]
                    final_img_green[i][j] = temp_green[image_green[i][j]]
                    final_img_blue[i][j] = temp_blue[image_blue[i][j]]

            final_img = cv2.merge((final_img_blue, final_img_green, final_img_red))

            self.image = final_img

            self.display_image2()
            self.t_label.setVisible(True)
            self.openbutton2.setVisible(True)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)

    def trans_histogram_matching(self):
        self.reset()
        if self.image is not None:
            self.image = self.tmp
            self.tmptarget = cv2.resize(self.tmptarget, (self.image.shape[0],self.image.shape[1]), interpolation=cv2.INTER_CUBIC)
            final_img_red = np.zeros((self.image.shape[0], self.image.shape[1]))
            final_img_green = np.zeros((self.image.shape[0], self.image.shape[1]))
            final_img_blue = np.zeros((self.image.shape[0], self.image.shape[1]))

            hist_img_red, hist_img_green, hist_img_blue, image_red, image_green, image_blue = self.histogram(self.image)

            e_hist_img_red, e_hist_img_green, e_hist_img_blue = self.cumulative_histogram(hist_img_red, hist_img_green,
                                                                                    hist_img_blue)

            final_hist_img_red = np.zeros((256,))
            final_hist_img_green = np.zeros((256,))
            final_hist_img_blue = np.zeros((256,))

            for x in range(256):
                final_hist_img_red[x] = 255 * (e_hist_img_red[x] - min(e_hist_img_red)) / (
                            max(e_hist_img_red) - min(e_hist_img_red))
                final_hist_img_green[x] = 255 * (e_hist_img_green[x] - min(e_hist_img_green)) / (
                            max(e_hist_img_green) - min(e_hist_img_green))
                final_hist_img_blue[x] = 255 * (e_hist_img_blue[x] - min(e_hist_img_blue)) / (
                            max(e_hist_img_blue) - min(e_hist_img_blue))

            thist_img_red, thist_img_green, thist_img_blue, timage_red, timage_green, timage_blue = self.histogram(self.tmptarget)

            te_hist_img_red, te_hist_img_green, te_hist_img_blue = self.cumulative_histogram(thist_img_red, thist_img_green,
                                                                                       thist_img_blue)

            tfinal_hist_img_red = np.zeros((256,))
            tfinal_hist_img_green = np.zeros((256,))
            tfinal_hist_img_blue = np.zeros((256,))
            for x in range(256):
                tfinal_hist_img_red[x] = 255 * (te_hist_img_red[x] - min(te_hist_img_red)) / (
                            max(te_hist_img_red) - min(te_hist_img_red))
                tfinal_hist_img_green[x] = 255 * (te_hist_img_green[x] - min(te_hist_img_green)) / (
                            max(te_hist_img_green) - min(te_hist_img_green))
                tfinal_hist_img_blue[x] = 255 * (te_hist_img_blue[x] - min(te_hist_img_blue)) / (
                            max(te_hist_img_blue) - min(te_hist_img_blue))

            temp_red = np.zeros((256,))
            temp_green = np.zeros((256,))
            temp_blue = np.zeros((256,))

            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_red[j] >= final_hist_img_red[i]:
                        temp_red[i] = j
                        break
            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_green[j] >= final_hist_img_green[i]:
                        temp_green[i] = j
                        break
            for i in range(256):
                for j in range(256):
                    if tfinal_hist_img_blue[j] >= final_hist_img_blue[i]:
                        temp_blue[i] = j
                        break

            image_red = np.array(image_red)
            image_green = np.array(image_green)
            image_blue = np.array(image_blue)
            image_red = image_red.reshape(self.image.shape[0], self.image.shape[1])
            image_green = image_green.reshape(self.image.shape[0], self.image.shape[1])
            image_blue = image_blue.reshape(self.image.shape[0], self.image.shape[1])

            for i in range(final_img_red.shape[0]):
                for j in range(final_img_red.shape[1]):
                    final_img_red[i][j] = temp_red[image_red[i][j]]
                    final_img_green[i][j] = temp_green[image_green[i][j]]
                    final_img_blue[i][j] = temp_blue[image_blue[i][j]]

            final_img = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=object)
            final_img = cv2.merge((final_img_blue, final_img_green, final_img_red))

            #cv2.imshow('test', final_img)
            cv2.waitKey()
            self.image = final_img

            self.display_image2()
            self.t_label.setVisible(True)
            self.openbutton2.setVisible(True)
            
            #Error calc
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)
    
    

    def Low_pass_filter(self):
        self.reset()
        hum, image_FShift = hum_calculate(self)
        GShift = image_FShift * hum
        image_G = np.fft.ifftshift(GShift)
        image_tranformed = np.abs(np.fft.ifft2(image_G))
        
        #self.image = image_tranformed
        
        cv2.imwrite('sample1.jpg', image_tranformed)
        self.image =cv2.imread('sample1.jpg')
        try: 
            os.remove("sample1.jpg")
        except: pass
        self.display_image(2)
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.transformButtonMatch.setVisible(False)
        
        ssim, self.image1 = ErrorMap(self.original_image,self.image)
        
        self.ssimVal.setText(str(ssim))
        self.display_image_error()
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.transformButtonMatch.setVisible(False)
        self.transformButtonShape.setVisible(False)
         
    

    def high_pass_filter(self):
        self.reset()
        hum, image_FShift = hum_calculate(self)
        hum_high = 1 - hum
        GShift = image_FShift * hum_high
        image_G = np.fft.ifftshift(GShift)
        image_tranformed_high = np.abs(np.fft.ifft2(image_G))
        
        #self.image = image_tranformed_high
        cv2.imwrite('sample1.jpg', image_tranformed_high)
        self.image =cv2.imread('sample1.jpg')
        try: 
            os.remove("sample1.jpg")
        except: pass
        self.display_image(2)
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.transformButtonMatch.setVisible(False)
        
        ssim, self.image1 = ErrorMap(self.original_image,self.image)
        
        self.ssimVal.setText(str(ssim))
        self.display_image_error()
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.transformButtonMatch.setVisible(False)
        self.transformButtonShape.setVisible(False)
        
    def band_pass_filter(self):
        self.reset()
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            
            #low pass
            hum, image_FShift = hum_calculate(self)
            GShift = image_FShift * hum
            image_G = np.fft.ifftshift(GShift)
            low_image = np.abs(np.fft.ifft2(image_G))
            
            #high pass
            hum, image_FShift = hum_calculate(self)
            hum_high = 1 - hum
            GShift = image_FShift * hum_high
            image_G = np.fft.ifftshift(GShift)
            high_image = np.abs(np.fft.ifft2(image_G))
            
            
            band_image = np.absolute(self.image - (low_image + high_image))
            #return  band_image
            cv2.imwrite('sample1.jpg', band_image)
            self.image =cv2.imread('sample1.jpg')
            try: 
                os.remove("sample1.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            self.transformButtonShape.setVisible(False)
    
    def FFT(self, x):
        length = len(x)
        if(length <= 1):
            return x
        even= self.FFT(x[0::2])
        odd = self.FFT(x[1::2])
        
        out = []
        for i in range(length//2):
            out.append(exp(-2j*pi*i/length)*odd[i])
            
        L = []
        R = []
        for i in range(length//2):
            L.append(even[i] + out[i])
            R.append(even[i] - out[i])
        
        return L + R
    
    def FFT_2D(self, image):
        result = []
        for i in range(image.shape[0]):
            result.append(self.FFT(image[i]))
        
        for i in range(len(result[0])):
            col = []
            for j in range(len(result)):
                col.append(result[j][i])
            col = self.FFT(col)
            
            for j in range(len(result)):
                result[j][i] = col[j]
                
        return result
    
    def fft_image(self):
        self.reset()
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            img_fft = self.FFT_2D(self.image)
            F = np.fft.fftshift((img_fft))
            plt.imshow(np.log1p(np.abs(F)), cmap = 'gray')
            plt.axis('off')
            plt.savefig('dft_image.jpg', bbox_inches='tight')
            #cv2.imwrite('sample1.jpg', output_img)
            self.image =cv2.imread('dft_image.jpg')
            try: 
                os.remove("dft_image.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            # ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            # self.ssimVal.setText(str(ssim))
            # self.display_image_error()
            # self.t_label.setVisible(False)
            # self.openbutton2.setVisible(False)
            # self.transformButtonMatch.setVisible(False) 

        
        
    def IFFT(self, x):
        length = len(x)
        if(length <= 1):
            return x
        
        even= self.IFFT(x[0::2])
        odd = self.IFFT(x[1::2])
        
        out = []
        for i in range(length//2):
            out.append(exp(2j*pi*i/length)*odd[i])
            
        L = []
        R = []
        for i in range(length//2):
            L.append(even[i] + out[i])
            R.append(even[i] - out[i])
            
        return L + R    
    
        
    def IFFT_2D(self, image):
        result = []
        
        for i in range(len(image)):
            result.append(self.IFFT(image[i]))
            
        for i in range(len(result[0])):
            col = []
            for j in range(len(result)):
                col.append(result[j][i])
            col = self.IFFT(col)
            
            for j in range(len(result)):
                result[j][i]=col[j]
                
        result = np.absolute(result)
        
        return result    
    
    def IFFT_image(self):
        self.reset()
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            img_ifft = self.IFFT_2D(self.FFT_2D(self.image))
            # Reconstucted image
            plt.imshow(img_ifft,cmap='gray')
            plt.axis('off')
            plt.savefig('reconstructed_image.png', bbox_inches='tight')
            self.image =cv2.imread('reconstructed_image.png')
            try: 
                os.remove("reconstructed_image.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
    def bilinear_interpolation(self):
        self.reset()
        if self.image is not None:
            self.image = self.original_image
            resize_scale_x = 2
            resize_scale_y = 2
            
            # Upsampled image size
            new_size_x = int(math.floor(self.image.shape[0]*resize_scale_x))
            new_size_y = int(math.floor(self.image.shape[1]*resize_scale_y))
            
            # Initialzing the output image
            output_img = np.zeros((new_size_x, new_size_y, self.image.shape[2]), dtype=np.uint8)
            
            # Computing the scaling factors for the x and y axes
            scale_x = float(self.image.shape[0] - 1) / float(new_size_x - 1)
            scale_y = float(self.image.shape[1] - 1) / float(new_size_y - 1)
            
            for y in range(new_size_y):
                
                for x in range(new_size_x):
                    
                    res = []
                
                    Xi = x * scale_x
                    Yi = y * scale_y
                
                    
                    mod_x = int(Xi)
                    mod_y = int(Yi)
                    x_floor = Xi - mod_x
                    y_floor = Yi - mod_y
                    x_plus = min(mod_x+1,self.image.shape[1]-1)
                    y_plus = min(mod_y+1,self.image.shape[0]-1)
        
            
                    for dim in range(self.image.shape[2]):
                    
                        bl = self.image[mod_y, mod_x, dim]
                        br = self.image[mod_y, x_plus, dim]
                        tl = self.image[y_plus, mod_x, dim]
                        tr = self.image[y_plus, x_plus, dim]
                    
                        b = x_floor * br + (1. - x_floor) * bl
                        t = x_floor * tr + (1. - x_floor) * tl
                        pixel = y_floor * t + (1. - y_floor) * b
                        res.append(int(pixel+0.5))
                        
                    output_img[y,x] = res
                
        #self.image = output_img
        cv2.imwrite('sample1.jpg', output_img)
        self.image =cv2.imread('sample1.jpg')
        try: 
            os.remove("sample1.jpg")
        except: pass
        self.display_image(2)
        self.t_label.setVisible(False)
        self.openbutton2.setVisible(False)
        self.transformButtonMatch.setVisible(False)
        
       
    
    
    def laplacian_filter(self):
        self.reset()
        """Initialzes and returns a 3X3 Laplacian filter"""
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
#         laplacian_filter = np.ones((3, 3), dtype='int')
            laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            laplacian_filter[laplacian_filter.shape[0] // 2][laplacian_filter.shape[1] // 2] = -8
            
            filtered_image = np.zeros((self.image.shape[0], self.image.shape[1]), dtype='float')
            row, col = laplacian_filter.shape[0], laplacian_filter.shape[1]
            image_temp = np.zeros((self.image.shape[0] + row - 1, self.image.shape[0] + col - 1), dtype='float')
            row //= 2
            col //= 2
            image_temp[row:image_temp.shape[0] - row, col:image_temp.shape[1] - col] = self.image
            for i in range(row, image_temp.shape[0] - row):
                for j in range(col, image_temp.shape[1] - col):
                    x = image_temp[i - row:i + row + 1, j - col:j + col + 1]
                    res = x * laplacian_filter
                    filtered_image[i - row][j - col] = res.sum()
            #self.image = filtered_image
            cv2.imwrite('sample1.jpg', filtered_image)
            self.image =cv2.imread('sample1.jpg')
            try: 
                os.remove("sample1.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
                
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
    
            
    def mean_filter(self):
        self.reset()
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            mean_image = np.zeros((self.image.shape),np.uint8)
            output = 0
            for i in range(1, self.image.shape[0]-1):
                    for j in range(1, self.image.shape[1]-1):
                        for m in range(-1, 2):
                            for n in range(-1, 2):
                                output += self.image[i+m, j+n]
                        mean_image[i,j] = int(output / (3*3))
                        output = 0
            #self.image = mean_image
            cv2.imwrite('sample1.jpg', mean_image)
            self.image = cv2.imread('sample1.jpg')
            try: 
                os.remove("sample1.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False) 
    
    
    def unsharp_mask_filter(self):
        self.reset()
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
            # transform image into frequency domain
            image_FShift = np.fft.fftshift(np.fft.fft2(self.image))
            # create low pass filter
            rows, cols = self.image.shape
            r = 50
            hum_unsharp = np.zeros((rows,cols), dtype = np.float32)
            for i in range(rows):
                for j in range(cols):
                    r_new = np.sqrt(((i-(rows/2))**2) + ((j - (cols/2))**2))
                    hum_unsharp[i,j] = np.exp(-r_new**2 / (2*r*r))
            GShift = image_FShift * hum_unsharp
            image_G = np.fft.ifftshift(GShift)
            image_tranformed_un = np.abs(np.fft.ifft2(image_G))
            mask = self.image - image_tranformed_un
            k = 1
            unmask_image = self.image + k * mask
            unmask_image = np.clip(unmask_image,0,255)
            
            #self.image = unmask_image
            cv2.imwrite('sample1.jpg', unmask_image)
            self.image = cv2.imread('sample1.jpg')
            try: 
                os.remove("sample1.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
    
    
    def median_filter(self):
        self.reset()
        if self.image is not None:
            #self.image = self.original_image
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            median_image = np.zeros((self.image.shape))
            kernal = [(0,0)]*(3*3)
            for i in range(self.image.shape[0]-1):
                for j in range(self.image.shape[1]-1):
                    kernal[0] = self.image[i-1,j-1]
                    kernal[1] = self.image[i-1,j]
                    kernal[2] = self.image[i-1,j+1]
                    kernal[3] = self.image[i,j-1]
                    kernal[4] = self.image[i,j]
                    kernal[5] = self.image[i,j+1]
                    kernal[6] = self.image[i+1,j-1]
                    kernal[7] = self.image[i+1,j]
                    kernal[8] = self.image[i+1,j+1]
                    
                    kernal.sort()
                    median_image[i,j] = kernal[4]

            #self.image=median_image
            cv2.imwrite('sample1.jpg', median_image)
            self.image = cv2.imread('sample1.jpg') 
            try: 
                os.remove("sample1.jpg")
            except: pass
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            ssim, self.image1 = ErrorMap(self.original_image,self.image)
            
            self.ssimVal.setText(str(ssim))
            self.display_image_error()
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False)
            
            
    def bicubic_interpolation(self):
        self.reset()
        if self.image is not None:
            self.image = self.original_image
            a = -0.5
            
            scale_factor = 2
            
            R, C, channel = self.image.shape
            
            # Getting padded image
            self.image = padding_img(self.image, R, C, channel)
            
            #Values for new image
            dR = math.floor(R*scale_factor)
            dC = math.floor(C*scale_factor)

            # Converting the image into matrix
            res = np.zeros((dR, dC, 3))
            
            v = 1/scale_factor

            for ch in range(channel):
                for r in range(dR):
                    for c in range(dC):
                        
                        x, y = c * v + 2, r * v + 2

                        x1 = 1 + x - math.floor(x)
                        x2 = x - math.floor(x)
                        x3 = math.floor(x) + 1 - x
                        x4 = math.floor(x) + 2 - x

                        y1 = 1 + y - math.floor(y)
                        y2 = y - math.floor(y)
                        y3 = math.floor(y) + 1 - y
                        y4 = math.floor(y) + 2 - y


                        matrix_k_x = np.matrix([[kernel(x1, a), kernel(x2, a), kernel(x3, a), kernel(x4, a)]])
                        
                        matrix_n = np.matrix([[self.image[int(y-y1), int(x-x1), ch],
                                            self.image[int(y-y2), int(x-x1), ch],
                                            self.image[int(y+y3), int(x-x1), ch],
                                            self.image[int(y+y4), int(x-x1), ch]],
                                        [self.image[int(y-y1), int(x-x2), ch],
                                            self.image[int(y-y2), int(x-x2), ch],
                                            self.image[int(y+y3), int(x-x2), ch],
                                            self.image[int(y+y4), int(x-x2), ch]],
                                        [self.image[int(y-y1), int(x+x3), ch],
                                            self.image[int(y-y2), int(x+x3), ch],
                                            self.image[int(y+y3), int(x+x3), ch],
                                            self.image[int(y+y4), int(x+x3), ch]],
                                        [self.image[int(y-y1), int(x+x4), ch],
                                            self.image[int(y-y2), int(x+x4), ch],
                                            self.image[int(y+y3), int(x+x4), ch],
                                            self.image[int(y+y4), int(x+x4), ch]]])
                        
                        matrix_k_y = np.matrix([[kernel(y1, a)], [kernel(y2, a)], [kernel(y3, a)], [kernel(y4, a)]])
                        res[r, c, ch] = np.dot(np.dot(matrix_k_x, matrix_n), matrix_k_y)


            # sys.stderr.write('\n')
            # sys.stderr.flush()
            
            cv2.imwrite('sample1.jpg', res)
            self.image = cv2.imread('sample1.jpg')  
            try: 
                os.remove("sample1.jpg")
            except: pass     
            self.display_image(2)
            self.t_label.setVisible(False)
            self.openbutton2.setVisible(False)
            self.transformButtonMatch.setVisible(False) 
            
            
            

    # Error Map and ssim generation code 

def ErrorMap(ref_img,transformed_img ):
    # Convert images to grayscale
    
    if len(ref_img.shape)>2:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img
    if len(transformed_img.shape)>2:
        transformed_gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    else:
        transformed_gray = transformed_img
    
   # Calculate the structural similarity (SSIM) index
    SSIM, diff = ssim(transformed_gray,ref_gray, full = True)
    diff = (diff * 255).astype("uint8")
    
    thresh_gamma = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts_gamma = cv2.findContours(thresh_gamma.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_gamma = imutils.grab_contours(cnts_gamma)
    
    return SSIM, thresh_gamma


def hum_calculate(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)  
            # transform image into frequency domain
            image_F = np.fft.fft2(self.image)
            image_FShift = np.fft.fftshift(image_F)
            # create low pass filter
            rows, cols = self.image.shape
            r = 50
            hum = np.zeros((rows,cols), dtype = np.float32)
            for i in range(rows):
                for j in range(cols):
                    r_new = np.sqrt(((i-(rows/2))**2) + ((j - (cols/2))**2))
                    if r_new <= r:
                        hum[i,j] = 1
                    else:
                        hum[i,j] = 0
            return hum, image_FShift
    

def padding_img( image, R, C,channel ):
    
    padded_img = np.zeros((R+4, C+4, channel))
    padded_img[2:R+2, 2:C+2, :channel] = image

    padded_img[2:R+2, 0:2, :channel] = image[:, 0:1, :channel]
    padded_img[R+2:R+4, 2:C+2, :] = image[R-1:R, :, :]
    padded_img[2:R+2, C+2:C+4, :] = image[:, C-1:C, :]
    padded_img[0:2, 2:C+2, :channel] = image[0:1, :, :channel]
    
    padded_img[0:2, 0:2, :channel] = image[0, 0, :channel]
    padded_img[R+2:R+4, 0:2, :channel] = image[R-1, 0, :channel]
    padded_img[R+2:R+4, C+2:C+4, :channel] = image[R-1, C-1, :channel]
    padded_img[0:2, C+2:C+4, :channel] = image[0, C-1, :channel]
    
    return padded_img


def kernel(x,s):
    
    if(np.abs(x)>=0)&(np.abs(x)<=1):
        return (s+2)*(np.abs(x)**3)-(s+3)*(np.abs(x)**2)+1
    
    if(np.abs(x)>1)&(np.abs(x)<=2):
        return s*(np.abs(x)**3)-(s*5)*(np.abs(x)**2)+((s*8)*np.abs(x))-s*4
    
    return 0





app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())
