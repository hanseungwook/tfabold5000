import os
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

# CONSTANTS
K_START = 100
K_END = 2000
TRIAL_NUM = 20
PATH = ''
FIGURE_PATH = '../figures'

def find_best_k(features_list_np, show_plot=False):
    k_list = np.linspace(K_START, K_END, num=TRIAL_NUM)

    silhouette_scores = []

    for k in k_list:
        result = KMeans(n_cluster=k, random_state=0).fit(features_list_np)
        labels = result.labels_

        # Silhouette score = [-1, 1]; -1 = incorrect clustering, 1 = highly dense clustering (well-separated)
        # Near 0 scores indicate overlapping clusters
        silhouette_scores.append(metrics.silhouette_score(features_list_np, labels, metric='euclidean'))

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlabel('K')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs. K')
    ax.plot(k_list, silhouette_scores)

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

    best_k = find_best_k(features_list_np, kwargs['show_plot'])

    print('K_START: {}, K_END: {} -> Best K: {}'.format(K_START, K_END, best_k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in VGG-extracted features as input')
    parser.add_argument('features_file', help='File containing VGG-extracted features')
    parser.add_argument('--show_plot', help='Flag to enable showing plot')
    args = parser.parse_args()
    main(**vars(args))




