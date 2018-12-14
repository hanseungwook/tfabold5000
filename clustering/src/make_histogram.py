#from sklearn.cluster import KMeans
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse
import math

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
sparse_shape = (8, 8, 8)

class ColorHistogram(object):
    def __init__(self, img_dir = None):
       self.img_dir = img_dir
       self.img_colors = []

    def load_images(self, n = None):
        self.img = []
        onlyfiles = [join(self.img_dir,f) for f in listdir(self.img_dir) if isfile(join(self.img_dir, f))]

        if n != None:
            onlyfiles = onlyfiles[:n]
        print(onlyfiles)

        clahe = cv.createCLAHE(clipLimit=40., tileGridSize=(16,16))
        for f in onlyfiles:
            img = cv.imread(f)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = img.reshape((img.shape[0] * img.shape[1], 3))
            #img = np.float32(img)
            img = cv.resize(img,(128,128))
            blue = img[:,:,0]
            green = img[:,:,1]
            red = img[:,:,2]

            blue = clahe.apply(blue)
            green = clahe.apply(green)
            red = clahe.apply(red)
            img = np.stack((blue,green,red),axis=-1)
            self.img.append(img)

    def histogram(self):
        counter = 0
        n = len(self.img)
        d = int(self.img[0].shape[0])
        print(d)
        
        for img in self.img:
            hist = np.zeros((4,8,4))
            for i in range(d):
                for j in range(d):
                    blue = (img[i,j,0]) >> 6
                    green = (img[i,j,1]) >> 5
                    red = (img[i,j,2]) >> 6

                    #print("blue: {}, green: {}, red: {}".format(blue,green,red))

                    hist[blue,green,red] += 1

            self.img_colors.append(hist)

            counter += 1
            print('Progress: %f' % (round(counter / n, 3) * 100))


    def save(self, out_file):
        np.save(out_file, self.img_colors)
        print('Saved to images.colors')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir",help="Path to image files")
    parser.add_argument("out_file",help="Output filename")
    args = parser.parse_args()

    ch = ColorHistogram(img_dir = args.img_dir)
    ch.load_images()
    ch.histogram()
    ch.save(args.out_file)

if __name__ == '__main__':
    main()
