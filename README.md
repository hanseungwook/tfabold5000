# Implementing TFA for BOLD 5000 dataset

## Dependencies

* Keras
* Numpy
* Sci-kit
* Tensorflow
* Pandas
* Seaborn
* Matplotlib



## How to run feature extraction


```{shell}
cd clustering/src/
python3 feature_extractor.py -m [Model-Name] -f [File-Path-of-Images]
python3 color_stats_extractor.py [img-path] [output-path]
```


### Model-Name Options
* VGG16
* VGG19
* InceptionV3
* ResNet50

## How to run kmeans clustering
```{shell}
cd clustering/src/
python3 kmeans.py [File-Path-of-Features]
```

The above program will save the figures to `../figures` and the labels/results to `../`.

## How to run cluster visualizer

```{shell}
cd clustering/src/
python3 cluster_visualizer.py [img_path] [label_path] [output-filename]
```

The visualizer above will run PCA on the original dataset with n_components = 2 to reduce the dimensionality of the dataset to 2 and save the visualized clusters to `../figures/[output-filename]`
