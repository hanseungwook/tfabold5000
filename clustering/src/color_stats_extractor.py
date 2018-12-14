import numpy as np
import cv2
import os
import argparse
from scipy.stats import skew
from cluster_visualizer import load_img

# Computes stats for each color component values
def compute_stats(c_values):
    c_mean = np.mean(c_values)
    c_std = np.std(c_values)
    c_skew = skew(c_values)

    return np.array([c_mean, c_std, c_skew])

# Loads R,G,B stat-based features
def load_features(data):
    n = len(data)
    features = np.zeros((n, 9))
    
    cnt = 0
    for img in data:
        r_values = np.array(img)[0::3]
        g_values = np.array(img)[1::3]
        b_values = np.array(img)[2::3]

        features[cnt][0:3] = compute_stats(r_values)
        features[cnt][3:6] = compute_stats(g_values)
        features[cnt][6:9] = compute_stats(b_values)
        
        cnt += 1

    return features

def main(**kwargs):
    img_path = kwargs['img_path']
    out_path = kwargs['out_path']
    data = load_img(img_path) # Returns list
    images = os.listdir(img_path)
    print('Images loaded.')

    features = load_features(data)
    print('Features loaded.')

    out_path = os.path.join(out_path, 'color_stats_features.npy')
    np.save(out_path, features)
    print('Features saved in {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='Path to image data set')
    parser.add_argument('out_path', help='Path to save the features')
    args = parser.parse_args()
    main(**vars(args))
