from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse

class ColorExtractor(object):
    def __init__(self, k = 5, img_dir = None):
       self.k = k
       self.img_dir = img_dir
       self.img_colors = []

    def load_images(self, n = None):
        self.img = []
        onlyfiles = [join(self.img_dir,f) for f in listdir(self.img_dir) if isfile(join(self.img_dir, f))]

        if n != None:
            onlyfiles = onlyfiles[:n]
        print(onlyfiles)

        for f in onlyfiles:
            img = cv.imread(f)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.reshape((img.shape[0] * img.shape[1], 3))
            self.img.append(img)

    def do_KMeans(self):
        for img in self.img:
            clt = KMeans(n_clusters = self.k)
            clt.fit(img)
            print(clt.cluster_centers_)
            self.img_colors.append(clt.cluster_centers_)

    def centroid_histogram(self, clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0,len(np.unique(clt.labels_))+1)
        (hist,_)=np.histogram(clt.labels_,bins=numLabels)
        hist=hist.astype("float")
        hist/=hist.sum()
        return hist

    def plot_colors(self, hist, centroids):
        bar = np.zeros((50,300,3), dtype="uint8")
        startX = 0
        for (percent,color) in zip(hist,centroids):
            endX = startX + (percent*300)
            cv.rectangle(bar,(int(startX),0),(int(endX),50),color.astype("uint8").tolist(),-1)
            startX=endX
        return bar

    def save(self):
        np.save('images_colors.npy', self.img_colors)
        print('Saved to images.colors')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k",help="Number of clusters")
    parser.add_argument("img_dir",help="Path to image files")
    args = parser.parse_args()

    ce = ColorExtractor(k = args.k, img_dir = args.img_dir)
    ce.load_images()
    clt = ce.do_KMeans()
    #hist = ce.centroid_histogram(clt)
    #bar = ce.plot_colors(hist,clt.cluster_centers_)
    #print(clt.cluster_centers_)
    ce.save()

    #plt.figure()
    #plt.axis("off")
    #plt.imshow(bar)
    #plt.show()

if __name__ == '__main__':
    main()
