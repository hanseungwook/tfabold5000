import os
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from progressbar import ProgressBar, ETA, Bar
import argparse
import multiprocessing as mp
import yaml

# CONSTANTS
K_START = 100
K_END = 500
TRIAL_NUM = 5
FIGURE_PATH = '../figures'
LABEL_PATH = '../labels'

def find_best_k(features_list_np, model, show_plot=False):
    k_list = np.linspace(K_START, K_END, num=TRIAL_NUM, dtype=np.int32)

    silhouette_scores = []

    print('{} cores found.'.format(mp.cpu_count()))

    trial = 1
    progress = ProgressBar(widgets=['Trial {}'.format(trial), ' ', pb.Percentage(), ' ',  Bar('='), ' ', ETA()])
    for k in progress(k_list):
        print('Clustering for K = {}...'.format(k))
        trial += 1
        # pool = mp.Pool(processes=mp.cpu_count())
        # print('{} Cores Found'.format(mp.cpu_count()))
        # result = pool.map(kmeans.fit, features_list_np)
        result = KMeans(n_clusters=k, random_state=0, n_jobs=-1, verbose=0).fit(features_list_np)
        labels = result.labels_

        if not os.path.isdir(LABEL_PATH):
            os.mkdir(LABEL_PATH)
        labelfile = open(os.path.join(LABEL_PATH,'{}_labels_{}'.format(model, k)), 'w')
        np.savetxt(labelfile, labels)
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
    ax_sil.plot(k_list, silhouette_scores)

    if not os.path.isdir(FIGURE_PATH):
        os.mkdir(FIGURE_PATH)

    # Save plot and Silhouette scores
    plt.savefig(os.path.join(FIGURE_PATH, '{}_silhouette_vs_k.png'.format(model)))
    outfile = open(os.path.join(FIGURE_PATH, '{}_results.txt'.format(model)), 'w+')
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

# Dimensions = (4916, 11, 11, 512)
def main(**kwargs):
    filename = kwargs['features_file']
    filepath = filename
    idx = filename.find('total_features')
    model = filename[idx+15:][:-4]
    features = np.load(filepath)
    features_list = []
    for i in range(features.shape[0]):
        features_list.append(features[i].flatten())

    features_list_np = np.array(features_list)
    print('Features loaded.')
    # print(features_list_np.shape)

    best_k = find_best_k(features_list_np, model, kwargs['show_plot'])

    print('K_START: {}, K_END: {} -> Best K: {}'.format(K_START, K_END, best_k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in VGG-extracted features as input')
    parser.add_argument('features_file', help='File containing VGG-extracted features')
    parser.add_argument('--show_plot', help='Flag to enable showing plot', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))
