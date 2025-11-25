# imports
import keras
from keras import layers
from keras import ops

import numpy as np
import math
import matplotlib.pyplot as plt
import os

# data preparation
num_classes = 100
input_shape = (3, 256, 256)

TempImageSet = "EarthRemoteSensingRapidResponse/TempImageSet"
# TODO: split images into train test based on folders, need to figure out x and y
# (x_train, y_train), (x_test, y_test) = 

# print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
# print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# NOTES:
# feed in 3d input (~, ~, ~)
# Transformer is treating input as 3 dimensional, filters should be 3d

# hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 1
num_epochs = 10 # use 100 for real training
image_size = 256 # input images are reszed to this
patch_size = 16  # size of patches to extract from input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
] # size of transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
] # size of dense layers for final classifier (temp: need to adjust)

# data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

# computer mean and variance of training data (for normalization)
# data_augmentation.layers[0].adapt(x_train)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        return 
        
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
       return
   
def create_vit_predictor():
    return