#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio
import pandas as pd
import seaborn as sns
import numpy as np
import argparse
import yaml
import os

class ClusterVisualizer:
    def __init__(self, img_path, label_path, out_file):
        self.img_path = img_path
        self.label_path = label_path
        self.labels = None
        self.out_file = out_file

    # Load image name and label from labels file
    # Returns dictionary (key = image name, value = label)
    def load_labels(self):
        print('Loading labels')
        with open(self.label_path, 'r') as labels:
            self.labels = dict(yaml.load(labels))
        
        return self.labels
    
    # Reads all the images and runs PCA
    # Returns dictionary (key = image name, value = PCA-ed data for image)
    def run_pca(self):
        img_list = []
        img_name_list = []

        print('Reading images')
        # Read all images
        for img in os.listdir(self.img_path):
            img_file = os.path.join(self.img_path, img)
            img_data = imageio.imread(img_file)
            img_list.append(img_data)
            img_name_list.append(img)

        print('Running PCA')
        # Run PCA
        pca = PCA(n_components=2)    
        img_pca = pca.fit_transform(img_list)

        img_pca_dict = dict(zip(img_name_list, img_pca))

        return img_pca_dict
    
    def plot(self, img_pca_dict):
        img_df = dict_2_df(img_pca_dict, self.labels, "Data", "Label")

        ax = sns.scatterplot(x='pca_1', y='pca_2', hue= img_df['Label'], data=img_df)
        #plt.show()
        plt.savefig(self.out_file)
        
    
def dict_2_df(dict1, dict2, col1, col2):
    df = pd.DataFrame({col1: pd.Series(dict1), col2: pd.Series(dict2)})
    df[['pca_1', 'pca_2']] = df['col1'].apply(pd.Series)

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
    parser.add_argument('img_path', help="Directory path for images")
    parser.add_argument('label_path', help="Labels for images")
    parser.add_argument('out_file', help="Output file for plot")
    args = parser.parse_args()

    cv = ClusterVisualizer(args.img_path, args.label_path, args.out_file)
    cv.load_labels()
    img_pca_dict = cv.run_pca()
    cv.plot(img_pca_dict)
    #test2()


if __name__ == "__main__":
    main()