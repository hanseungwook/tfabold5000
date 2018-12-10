import os
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from progressbar import ProgressBar, ETA, Bar, Timer
# from collections import defaultdict
import argparse
import multiprocessing as mp
import yaml

# CONSTANTS
K_START = 2
K_END = 10
TRIAL_NUM = 9
IMAGE_PATH = '../../bold5000-dataset/scene'
FIGURE_PATH = '../figures'
LABEL_PATH = '../labels'

def find_best_k(images, features_list_np, color_k, show_plot=False):
    k_list = np.linspace(K_START, K_END, num=TRIAL_NUM, dtype=np.int32)

    silhouette_scores = []

    print('{} cores found.'.format(mp.cpu_count()))

    trial = 1
    progress = ProgressBar(widgets=[pb.Percentage(), ' ',  Bar('='), ' ', Timer()]).start()
    for k in progress(k_list):
        print('Clustering for K = {}...'.format(k))
        trial += 1
        # pool = mp.Pool(processes=mp.cpu_count())
        # print('{} Cores Found'.format(mp.cpu_count()))
        # result = pool.map(kmeans.fit, features_list_np)
        result = KMeans(n_clusters=k, random_state=0, n_jobs=-1, verbose=0).fit(features_list_np)
        labels = list(map(int, result.labels_))
        image_label_pairs = list(zip(images, labels))

        if not os.path.isdir(LABEL_PATH):
            os.mkdir(LABEL_PATH)
        labelfile = open(os.path.join(LABEL_PATH,'ck{}_labels_{}.yml'.format(color_k, k)), 'w')
        #np.savetxt(labelfile, image_label_pairs)
        yaml.dump(image_label_pairs, labelfile, default_flow_style=False)
        labelfile.close()

        # Silhouette score = [-1, 1]; -1 = incorrect clustering, 1 = highly dense clustering (well-separated)
        # Near 0 scores indicate overlapping clusters
        silhouette_scores.append(metrics.silhouette_score(features_list_np, labels, metric='euclidean'))

    print('Generating Silhouette Score vs. K Plot')
    # Plot Silhouette Score vs. K
    figure_sil = plt.figure()
    ax_sil = figure_sil.add_subplot(1, 1, 1)
    ax_sil.set_xlabel('K')
    ax_sil.set_ylabel('Silhouette Score')
    ax_sil.set_title('Silhouette Score vs. K')
    ax_sil.scatter(k_list, silhouette_scores)

    if not os.path.isdir(FIGURE_PATH):
        os.mkdir(FIGURE_PATH)

    # Save plot and Silhouette scores
    plt.savefig(os.path.join(FIGURE_PATH, 'ck{}_silhouette_vs_k.png'.format(color_k)))
    outfile = open(os.path.join(FIGURE_PATH, 'ck{}_silhouette_vs_k.txt'.format(color_k)), 'w+')
    results = list(zip(k_list, silhouette_scores))
    #yaml.dump(results, outfile, default_flow_style=True)
    np.savetxt(outfile, results)
    outfile.close()

    if show_plot:
        plt.show()

    best_k = None
    best_score = float('-inf')
    for result in results:
        if best_score < result[1]:
            best_score = result[1]
            best_k = result[0]

    return best_k

def sample_from_cluster(labelfile):
    pass

# Dimensions = (4916, 11, 11, 512)
# Dominant color dimensions = (1000, 5, 3)
def main(**kwargs):
    imagepath = kwargs['image_path']
    filepath = kwargs['features_file']
    #idx = filename.find('total_features')
    #model = filename[idx+15:][:-4]
    idx1 = filepath.find('images_color')
    idx2 = filepath.find('.npy')
    color_k = filepath[idx1+12:idx2]
    images = os.listdir(imagepath)
    features = np.load(filepath)
    features_list = []
    for i in range(features.shape[0]):
        features_list.append(features[i].flatten())

    features_list_np = np.array(features_list)
    print('Features loaded.')
    # print(features_list_np.shape)

    best_k = find_best_k(images, features_list_np, color_k, kwargs['show_plot'])

    print('K_START: {}, K_END: {} -> Best K: {}'.format(K_START, K_END, best_k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in VGG-extracted features as input')
    parser.add_argument('image_path', help='Path to image data')
    parser.add_argument('features_file', help='File containing features')
    parser.add_argument('--show_plot', help='Flag to enable showing plot', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))
