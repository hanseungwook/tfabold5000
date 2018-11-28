import os
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar, ETA, Bar
import argparse
import yaml

# CONSTANTS
K_START = 100
K_END = 2000
TRIAL_NUM = 20
PATH = '../vgg_features'
FIGURE_PATH = '../figures'

def find_best_k(features_list_np, show_plot=False):
    k_list = np.linspace(K_START, K_END, num=TRIAL_NUM, dtype=np.int64)

    silhouette_scores = []

    trial = 1
    progress = ProgressBar(widgets=['Trial {}'.format(trial), Bar('='), ETA()])
    for k in progress(k_list):
        print('Clustering for K = {}...'.format(k))
        trial += 1
        result = KMeans(n_clusters=k, random_state=0).fit(features_list_np)
        labels = result.labels_

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
    ax_sil.plot(k_list, silhouette_scores)

    if not os.path.isdir(FIGURE_PATH):
        os.mkdir(FIGURE_PATH)

    # Save plot and Silhouette scores
    plt.savefig(os.path.join(FIGURE_PATH), 'silhouette_vs_k.png')
    outfile = open(os.path.join(FIGURE_PATH, 'results.txt'), 'w+')
    results = zip(k_list, silhouette_scores)
    yaml.dump(outfile, results)

    if show_plot:
        plt.show()

    for i in range(len(results)):
        best_k = None
        best_score = float('-inf')
        if best_k < results[i][1]:
            best_score = results[i][1]
            best_k = results[i][0]

    return best_k

# Dimensions = (4916, 11, 11, 512)
def main(**kwargs):
    filename = kwargs['features_file']
    filepath = os.path.join(PATH, filename)
    features = np.load(filepath)
    features_list = []
    for i in range(features.shape[0]):
        features_list.append(features[i].flatten())

    features_list_np = np.array(features_list)
    print('Features loaded.')
    print(features_list_np.shape)

    best_k = find_best_k(features_list_np, kwargs['show_plot'])

    print('K_START: {}, K_END: {} -> Best K: {}'.format(K_START, K_END, best_k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in VGG-extracted features as input')
    parser.add_argument('features_file', help='File containing VGG-extracted features')
    parser.add_argument('--show_plot', help='Flag to enable showing plot', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))