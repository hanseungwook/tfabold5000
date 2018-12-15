# Implementing TFA for BOLD 5000 dataset

## Dependencies

* Keras
* Numpy
* Sci-kit
* Tensorflow
* Pandas
* Seaborn
* Matplotlib
* BrainIAK
* ggplot


## How to run feature extraction using pre-trained CNN

```{shell}
cd clustering/src/
python3 feature_extractor.py -m [Model-Name] -f [File-Path-of-Images]
```

This will write the features extracted using the specified deep convolutional neural network to the current directory.

### Model-Name Options
* VGG16
* VGG19
* InceptionV3
* ResNet50

## How to run color extraction
### Feature = (Mean, SD, Skew)
```{shell}
python3 color_stats_extractor.py [img-path] [output-path]
```

This will read the original images to construct the RGB distribution features of each image and write the features to the [output-path]

## How to run kmeans clustering
To run K-Means on the features:
```{shell}
cd clustering/src/
python3 kmeans.py [Image-Path] [File-Path-of-Features] [Prefix-of-Figure/Label-Names] [--pca=pca_n / optional] [--show_plot / optional]
```

The above program runs kmeans clustering algorithm on the features of the images and will save the figures to `../figures` and the labels/results to `../`.

To run K-means on the images themselves:
```{shell}
python3 kmeans.py [Image-Path] [--pca=pca_n / optional] [--show_plot / optional]
```

The above command will run kmeans clustering algorithm on the images in the specified image-path above.

It will create the silhouette score list and plots in ../figures (respect to the location of the program kmeans.py) and labels/clusters yaml files in ../labels. 



## How to run cluster visualizer

```{shell}
cd clustering/src/
python3 cluster_visualizer.py [image_path/feature_path] [label_path] [output-filename]
```

The visualizer above will run PCA on the given image or feature dataset with n_components = 2 to reduce the dimensionality of the dataset to 2 and save the visualized clusters to `../figures/[output-filename]`

## How to run make_histogram.py

```{shell}
cd clustering/src/
python3 make_histogram.py [image_path] [output-filename]
```

This program will extract RGB Histogram features from the original images and save the features in the current directory under the [output-filename].

## How to run tfa bold
```{shell}
cd tfa/
python3 tfa_bold.py json_file [--K number_of_hubs_to_locate] [--n number_of_iterations] [--voxel] [--tfa]
```
