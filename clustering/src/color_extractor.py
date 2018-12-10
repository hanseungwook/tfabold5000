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

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

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
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = img.reshape((img.shape[0] * img.shape[1], 3))
            #img = np.float32(img)
            self.img.append(img)

    def do_KMeans(self, q = None):
        counter = 0
        n = len(self.img)
        
        for img in self.img:
            if q == None:
                clt = KMeans(n_clusters = self.k, n_jobs=-1)
                clt.fit(img)
                self.img_colors.append(clt.cluster_centers_)
            else:
                Z = img.reshape((-1,3))
                Z = np.float32(Z)

                ret,label,center=cv.kmeans(Z,self.k,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((img.shape))
                res2 = cv.resize(res2,(100,100))
                self.img_colors.append(res2)


            #print(clt.cluster_centers_)
            counter += 1
            print('Progress: %f' % (round(counter / n, 3) * 100))

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

    def save(self, out_file):
        np.save('../features/' + out_file, self.img_colors)
        print('Saved to images.colors')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k",help="Number of clusters")
    parser.add_argument("img_dir",help="Path to image files")
    parser.add_argument("out_file",help="Output filename")
    args = parser.parse_args()

    ce = ColorExtractor(k = int(args.k), img_dir = args.img_dir)
    ce.load_images(n = 1)
    clt = ce.do_KMeans(q=1)
    #hist = ce.centroid_histogram(clt)
    #bar = ce.plot_colors(hist,clt.cluster_centers_)
    #print(clt.cluster_centers_)
    ce.save(args.out_file)

    #plt.figure()
    #plt.axis("off")
    #plt.imshow(bar)
    #plt.show()
    '''
    img = cv.imread('/Users/hyundonglee/Desktop/CNB/tfabold5000/clustering/images/bold5000-dataset/scene/exerciseequipment3.jpg')
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv.resize(res2,(100,100))

    cv.imshow('res2',res2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()
