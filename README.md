# Implementing TFA for BOLD 5000 dataset

## Dependencies

* Keras
* Numpy
* Sci-kit
* Tensorflow




## How to run feature extraction


```{shell}
cd clustering/src/
python3 feature_extractor.py -m [Model-Name] -f [File-Path-of-Images]
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
