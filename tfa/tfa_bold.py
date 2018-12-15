from brainiak.factoranalysis.tfa import TFA
from brainiak.factoranalysis.htfa import HTFA

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import hypertools as hyp
import seaborn as sns

import nilearn.plotting as niplot
from nilearn.masking import apply_mask
from nilearn import image

from nilearn.input_data import NiftiMasker
import nibabel as nib
import numpy as np
import scipy.spatial.distance as sd

import os, sys, warnings, json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from copy import copy as copy

from itertools import combinations
import random

from os import listdir
from os.path import isfile, join
from collections import defaultdict

import argparse
import yaml


def cmu2nii(Y, R, template):
    Y = np.array(Y, ndmin=2)
    img = template
    S = img.affine
    locs = np.array(np.dot(R - S[:3, 3], np.linalg.inv(S[0:3, 0:3])), dtype='int')
    data = np.zeros(tuple(list(img.shape)[0:3]+[Y.shape[0]]))
    for i in range(Y.shape[0]):
        for j in range(R.shape[0]):
            data[locs[j, 0], locs[j, 1], locs[j, 2], i] = Y[i, j]
    return nib.Nifti1Image(data, affine=img.affine)

# data formatting: .nii --> matrix format
def nii2cmu(nifti_file, mask_file=None):
    S = nifti_file.get_sform()
    mask = nib.load('/Users/hyundonglee/Desktop/Nifti/'+mask_file)
    Y = apply_mask(nifti_file, mask)
    #np.save('Y.npy',Y)

    coords = np.nonzero(mask.dataobj)
    i = np.nonzero(mask.dataobj)[0]
    j = np.nonzero(mask.dataobj)[1]
    k = np.nonzero(mask.dataobj)[2]
    ijk = np.vstack((i,j,k))
    ijk1 = np.vstack((ijk,np.ones(i.shape)))
    affine = np.dot(S,ijk1)
    affine = affine.T
    affine3 = affine[:,:-1]

    return {'Y': Y, 'R': affine3}

def dynamic_ISFC(data, windowsize=0):
    """
    :param data: a list of number-of-observations by number-of-features matrices
    :param windowsize: number of observations to include in each sliding window (set to 0 or don't specify if all
                       timepoints should be used)
    :return: number-of-features by number-of-features isfc matrix

    reference: http://www.nature.com/articles/ncomms12141
    """

    def rows(x): return x.shape[0]
    def cols(x): return x.shape[1]
    def r2z(r): return 0.5*(np.log(1+r) - np.log(1-r))
    def z2r(z): return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)
    
    def vectorize(m):
        np.fill_diagonal(m, 0)
        return sd.squareform(m)
    
    assert len(data) > 1
    
    ns = list(map(rows, data))
    vs = list(map(cols, data))

    n = np.min(ns)
    if windowsize == 0:
        windowsize = n

    assert len(np.unique(vs)) == 1
    v = vs[0]

    isfc_mat = np.zeros([n - windowsize + 1, int((v ** 2 - v)/2)])
    for n in range(0, n - windowsize + 1):
        next_inds = range(n, n + windowsize)
        for i in range(0, len(data)):
            mean_other_data = np.zeros([len(next_inds), v])
            for j in range(0, len(data)):
                if i == j:
                    continue
                mean_other_data = mean_other_data + data[j][next_inds, :]
            mean_other_data /= (len(data)-1)
            next_corrs = np.array(r2z(1 - sd.cdist(data[i][next_inds, :].T, mean_other_data.T, 'correlation')))            
            isfc_mat[n, :] = isfc_mat[n, :] + vectorize(next_corrs + next_corrs.T)
        isfc_mat[n, :] = z2r(isfc_mat[n, :]/(2*len(data)))
    
    isfc_mat[np.where(np.isnan(isfc_mat))] = 0
    return isfc_mat


def main():
    parser = argparse.ArgumentParser(description='Input: cluster info, Output: brain images')
    parser.add_argument("json_file", help='cluster information: key=cluster_number, value=name of images')
    parser.add_argument("--K", type=int, default=5, help='number of points on the brain')
    parser.add_argument("--n", type=int, default=0, help='number of combinations')
    parser.add_argument("--voxel")
    parser.add_argument("--tfa")
    args = parser.parse_args()

    json_file = args.json_file
    out_dir = json_file.strip('.yml')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    K = args.K
    n = args.n
    voxel = args.voxel

    # read in image name to fmri data mapping info
    onlyfiles = [join('/Users/hyundonglee/Desktop/Order/',f) for f in listdir('/Users/hyundonglee/Desktop/Order') if isfile(join('/Users/hyundonglee/Desktop/Order/', f)) and '.txt' in f]
    
    fmri = defaultdict(list)

    for f in onlyfiles:
        with open(f, 'rb') as fh:
            name = f.split('.')[0]
            i = 3
            for line in fh:
                line = line.strip()
                fn = name.split('/')[-1].split('sess')[0] + 'ses-' + name.split('/')[-1].split('sess')[1].split('run')[0] + 'run-' + name.split('/')[-1].split('sess')[1].split('run')[1]
                if 'CSI_' in fn:
                    fn = fn.split('CSI_')[0] + 'CSI1_' + fn.split('CSI_')[1]
                fmri[line].append(('sub-'+fn+'_bold_stnd.nii',i,i+3,'sub-'+fn+'_brainmask.nii'))
                i += 5

    with open(json_file,'r') as fh:
        #cluster = json.load(fh)
        cluster = yaml.load(fh)

    i = 0
    for pair in list(combinations(cluster.keys(), 2)):
        cluster_x = pair[0]
        cluster_y = pair[1]

        # pick 1 image from each cluster
        # retrieve relevant fmri data for x,y
        for i in range(10):
            random.shuffle(cluster[cluster_x])
        for i in range(5):
            random.shuffle(cluster[cluster_y])
        for x in cluster[cluster_x]:
            fmri_x = fmri[bytes(x, encoding='utf-8')] # (*_bold_stnd.nii, slice_start, slice_end, *_brainmask.nii)
            if len(fmri_x) > 0:
                x_n = x.split('.')[0]
                break

        for y in cluster[cluster_y]:
            fmri_y = fmri[bytes(y, encoding='utf-8')] # (*_bold_stnd.nii, slice_start, slice_end, *_brainmask.nii)
            if len(fmri_y) > 0:
                if fmri_y[0][0] == fmri_x[0][0]:
                    continue
                y_n = y.split('.')[0]
                break

        #fmri_x = fmri[bytes('rep_homeoffice9.jpg', encoding='utf-8')][0]
        #fmri_y = fmri[bytes('rep_homeoffice9.jpg', encoding='utf-8')][1]
        #x_n = y_n = 'rep_homeoffice9'

        fmri_x = fmri_x[random.randint(0,len(fmri_x)-1)]
        fmri_y = fmri_y[random.randint(0,len(fmri_y)-1)]
        subj_x = fmri_x[0].split('_bold')[0].lstrip('sub-') + '_'
        subj_y = fmri_y[0].split('_bold')[0].lstrip('sub-') + '_'

        x_nii = image.smooth_img('/Users/hyundonglee/Desktop/Nifti/'+fmri_x[0], fwhm=7)
        y_nii = image.smooth_img('/Users/hyundonglee/Desktop/Nifti/'+fmri_y[0], fwhm=7) 

        cmu_data = list(map(lambda n: nii2cmu(n, fmri_x[3]), [x_nii.slicer[:,:,:,fmri_x[1]:fmri_x[2]]]))
        cmu_data.extend(list(map(lambda n: nii2cmu(n, fmri_y[3]), [y_nii.slicer[:,:,:,fmri_y[1]:fmri_y[2]]])))

        if args.voxel:
            print("saving voxel_locations")
            hyp.plot(cmu_data[0]['R'], 'k.', save_path=out_dir+'/voxel_locations_'+subj_x+x_n+'.png')
            hyp.plot(cmu_data[1]['R'], 'k.', save_path=out_dir+'/voxel_locations_'+subj_y+y_n+'.png')
            print("save voxel_locations complete")

        # Convert between (timepoints by voxels) and (voxels by timepoints) data matrices
        htfa_data = list(map(lambda x: {'R': x['R'], 'Z': x['Y'].T}, cmu_data))
        nvoxels, ntimepoints = htfa_data[0]['Z'].shape

        if args.tfa:
            # Use TFA to find network hubs in one subject's data
            print("running TFA")
            tfa_x = TFA(K=K, max_num_voxel=int(nvoxels*0.05),
                    max_num_tr = int(ntimepoints), verbose=False)
            tfa_x.fit(htfa_data[0]['Z'], htfa_data[0]['R'])

            #plot the hubs on a glass brain!
            niplot.plot_connectome(np.eye(K),
                    tfa_x.get_centers(tfa_x.local_posterior_),
                    node_color='k',
                    output_file=out_dir+'/network_hubs_'+subj_x+x_n+'.png')

            # Visualizing how the brain images are simplified using TFA
            original_image = cmu2nii(htfa_data[0]['Z'][:, 0].T, htfa_data[0]['R'], x_nii)
            niplot.plot_glass_brain(original_image, plot_abs=False,
                    output_file=out_dir+'/simplified_by_TFA_'+subj_x+x_n+'.png')

            connectome = 1 - sd.squareform(sd.pdist(tfa_x.W_), 'correlation')
            niplot.plot_connectome(connectome,
                    tfa_x.get_centers(tfa_x.local_posterior_),
                    node_color='k',
                    edge_threshold='75%',
                    output_file=out_dir+'/connectome'+subj_x+x_n+'.png')

            tfa_y = TFA(K=K, max_num_voxel=int(nvoxels*0.05),
                    max_num_tr = int(ntimepoints), verbose=False)
            tfa_y.fit(htfa_data[1]['Z'], htfa_data[1]['R'])

            #plot the hubs on a glass brain!
            niplot.plot_connectome(np.eye(K),
                    tfa_y.get_centers(tfa_y.local_posterior_),
                    node_color='k',
                    output_file=out_dir+'/network_hubs_'+subj_y+y_n+'.png')

            # Visualizing how the brain images are simplified using TFA
            original_image = cmu2nii(htfa_data[1]['Z'][:, 0].T, htfa_data[1]['R'], y_nii)
            niplot.plot_glass_brain(original_image, plot_abs=False,
                    output_file=out_dir+'/simplified_by_TFA_'+subj_y+y_n+'.png')

            connectome = 1 - sd.squareform(sd.pdist(tfa_y.W_), 'correlation')
            niplot.plot_connectome(connectome,
                    tfa_y.get_centers(tfa_y.local_posterior_),
                    node_color='k',
                    edge_threshold='75%',
                    output_file=out_dir+'/connectome'+subj_y+y_n+'.png')

            print("TFA complete")

        print("running HTFA")
        htfa = HTFA(K=K,n_subj=len(htfa_data),
                max_global_iter=5,
                max_local_iter=2,
                voxel_ratio=0.5,
                tr_ratio=0.5,
                max_voxel=int(nvoxels*0.05),
                max_tr=int(ntimepoints))

        htfa.fit(list(map(lambda x: x['Z'], htfa_data)),
                list(map(lambda x: x['R'], htfa_data)))

        #set the node display properties
        colors = np.repeat(np.vstack([[0, 0, 0],
            sns.color_palette("Spectral", htfa.n_subj)]), K, axis=0)
        colors = list(map(lambda x: x[0],
            np.array_split(colors, colors.shape[0], axis=0))) #make colors into a list
        sizes = np.repeat(np.hstack([np.array(50), np.array(htfa.n_subj*[20])]), K)

        #extract the node locations from the fitted htfa model
        global_centers = htfa.get_centers(htfa.global_posterior_)
        local_centers = list(map(htfa.get_centers,
            np.array_split(htfa.local_posterior_, htfa.n_subj)))
        centers = np.vstack([global_centers, np.vstack(local_centers)])

        #make the plot
        niplot.plot_connectome(np.eye(K*(1 + htfa.n_subj)),
                centers, node_color=colors,
                node_size=sizes,
                output_file=out_dir+'/htfa_'+subj_x+x_n+'_'+subj_y+y_n+'.png')


        n_timepoints = list(map(lambda x: x['Z'].shape[1], htfa_data)) #number of timepoints for each person
        inds = np.hstack([0, np.cumsum(np.multiply(K, n_timepoints))])
        W = list(map(lambda i: htfa.local_weights_[inds[i]:inds[i+1]].reshape([K, n_timepoints[i]]).T,
            np.arange(htfa.n_subj)))

        static_isfc = dynamic_ISFC(W)
        niplot.plot_connectome(sd.squareform(static_isfc[0, :]),
                global_centers,node_color='k',
                edge_threshold='75%',
                output_file=out_dir+'/static_isfc_'+subj_x+x_n+'_'+subj_y+y_n+'.png')


        print("HTFA complete")

        i += 1
        if n != 0 and i == n:
            break

if __name__ == '__main__':
    main()
