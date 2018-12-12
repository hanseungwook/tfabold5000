#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import *
import imageio
import pandas as pd
import seaborn as sns
import numpy as np
import argparse
import yaml
import os

class ClusterVisualizer:
    def __init__(self, feature_path, label_path, out_file):
        #self.img_path = img_path
        self.feature_path = feature_path
        self.label_path = label_path
        self.labels = None
        self.out_file = out_file

    # Load image name and label from labels file
    # Returns dictionary (key = image name, value = label)
    def load_labels(self):
        print('Loading labels')
        with open(self.label_path, 'r') as labels:
            labels_list = yaml.load(labels)
        
        self.img_name_list = [label[0] for label in labels_list]
        self.labels = dict(labels_list)
        
        return self.labels
    
    # Reads all the images and runs PCA
    # Returns dictionary (key = image name, value = PCA-ed data for image)
    def run_pca(self, pca_n=2):
        img_list = []

        print('Reading images')
        """
        # Read all images
        for img in os.listdir(self.img_path):
            img_file = os.path.join(self.img_path, img)
            img_data = imageio.imread(img_file)
            img_list.append(img_data.flatten())
        """
        # Read features of images
        features = np.load(self.feature_path)
        features_flat = [feature.ravel() for feature in features]
        img_list = features_flat
       
        # Checking dimensions of images
        """
        size = None
        counter = 0
        for img in img_list:
            if not size:
                size = len(img)
            else:
                if size != len(img):
                    print(img_name_list[counter])
                    raise Exception("error")

            counter += 1
        """

        print('Running PCA')
        # Run PCA
        pca = PCA(n_components=pca_n)    
        img_pca = pca.fit_transform(img_list)

        #img_pca_dict = dict(zip(self.img_name_list, img_pca))

        return img_pca
    
    def plot(self, img_pca_dict):
        img_df = dict_2_df(img_pca_dict, self.labels, "Data", "Label")

        ax = sns.scatterplot(x='comp_1', y='comp_2', hue= img_df['Label'], data=img_df)
        #plt.show()
        plt.savefig(os.path.join('../figures/', self.out_file))

    # def plot_tsne(self, img_tsne_dict):
    #     img_df = dict_2_df(img_tsne_dict, self.labels, "Data", "Label")
    #     ax = sns.scatterplot(x='comp_1', y='comp_2', hue= img_df['Label'], data=img_df)

    def run_tsne(self, img_pca):
        tsne = TSNE(n_components=2, verbose=1)
        img_tsne = tsne.fit_transform(img_pca)
        img_tsne_dict = dict(zip(self.img_name_list, img_tsne))
        
        return img_tsne_dict
        
        
    
def dict_2_df(dict1, dict2, col1, col2):
    df = pd.DataFrame({col1: pd.Series(dict1), col2: pd.Series(dict2)})
    df[['comp_1', 'comp_2']] = df[col1].apply(pd.Series)

    return df

# Testing dict_2_df
def test1():
    dict1 = {"a": (1,2), "b": 1}
    dict2 = {"a": (3,4), "b": 0}
    print(dict_2_df(dict1, dict2, "col1", "col2"))

def test2():
    dict1 = {"a": (1,2), "b": (3,4)}
    dict2 = {"a": 1, "b": 0}
    df = dict_2_df(dict1, dict2, "col1", "col2")
    print(df)
    ax = sns.scatterplot(x='pca_1', y = 'pca_2', hue = df['col2'], data=df)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize clusters in iamges dir with respective cluster")
    #parser.add_argument('img_path', help="Directory path for images")
    parser.add_argument('feature_path', help="Directory path for features of images")
    parser.add_argument('label_path', help="Labels for images")
    parser.add_argument('out_file', help="Output file for plot")
    args = parser.parse_args()

    #cv = ClusterVisualizer(args.img_path, args.label_path, args.out_file)
    cv = ClusterVisualizer(args.feature_path, args.label_path, args.out_file)
    cv.load_labels()
    img_pca_dict = cv.run_pca()
    cv.plot(img_pca_dict)
    #test2()


if __name__ == "__main__":
    main()
