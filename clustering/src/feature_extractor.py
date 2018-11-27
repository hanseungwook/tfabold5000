import os
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import numpy as np
import argparse


class FeatureExtractor:
    def __init__(self, __model, __filepath):
        self.create_model(__model)
        self.filepath = __filepath
        self.total_features = []
    
    def create_model(self, model):
        if model == "VGG16":
            self.model = VGG16(weights='imagenet', include_top=False)
        elif model == "VGG19":
            self.model = VGG19(weights="imagenet", include_top=False)
        elif model == "ResNet50":
            self.model = ResNet50(weights="imagenet", include_top=False)
        elif model == "InceptionV3":
            self.model = InceptionV3(weights='imagenet', include_top=False)
        else:
            raise Exception("Incorrect model for feature extraction specified")
        
    def extract(self):
        counter = 0
        for img in os.listdir(self.filepath):
            img_path = os.path.join(self.filepath, img)
            img = image.load_img(img_path, target_size=(375, 375))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            feature = self.model.predict(img_data)
            print(counter)
            counter += 1
            #print(feature)

            self.total_features.append(feature)
        
        print(len(self.total_features))
        
    def save_features(self):
        if len(self.total_features) <= 0:
            raise Exception("No features extracted")
        total_features_save = np.array(self.total_features)
        np.savetxt("extracted-feature-dataset.txt", total_features_save)

    def save_image_names(self):
        with open(os.path.join('.', "image-names.txt"), "w") as outputFile:
            for img in os.listdir(self.filepath):
                outputFile.write(img)
                outputFile.write('\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model and filepath")
    parser.add_argument('-m', choices=['VGG16', 'VGG19', 'InceptionV3', 'ResNet50'], required=True)
    parser.add_argument('-f', required=True)
    args = parser.parse_args()

    fe = FeatureExtractor(args.m, args.f)
    #fe.extract()
    #fe.save_features()
    fe.save_image_names()
    

    