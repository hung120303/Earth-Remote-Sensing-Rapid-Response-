#################################################################################################################################
# ERSRR_Model.py                                                                                                                #
#   - This program file handles training the model, evaluating its accuracy and prediction tasks using the generated model.     #
#     The implementation of the model is derived from Khalid Salama's keras implemenation of the vision transformer model       #
#     proposed in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", modified into a encoder/decoder #
#     model for plume detection.                                                                                                #
#       - https://keras.io/examples/vision/image_classification_with_vision_transformer/                                        #
#       - https://github.com/tuvovan/Vision_Transformer_Keras                                                                   #
#       - https://doi.org/10.48550/arXiv.2010.11929                                                                             #
#################################################################################################################################

# imports
import keras
from keras import layers
from keras import ops
from sklearn.model_selection import train_test_split 
import simplekml
import rasterio
from rasterio.plot import show
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# data preparation
num_classes = 100
input_shape = (256, 256, 5)
output_shape = (256, 256, 1)

# hyperparameters
learning_rate = 0.01
weight_decay = 0.001
batch_size = 2
num_epochs = 5
image_size = 256 # input is resized to (image_size x image_size) 
patch_size = 16  # size of patches to extract from input
num_patches = (image_size // patch_size) ** 2
projection_dim = 64 # set to 64 for small datasets, otherwise 768 or 1024
num_heads = 4
upscale_factor = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
] # size of transformer layers
transformer_layers = 4
conv_layers = 2
mlp_head_units = [
    2048,
    1024,
] # size of dense layers for final classifier (temp: need to adjust)

####################################################
# class Patches                                    #
#    - implementation of patch creation as a layer #
# Methods:                                         #
#    - call(self, images)                          #
#    - get_config(self)                            #
####################################################
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        bands = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * bands,
            ),
        )
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
        
####################################################
# class PatcheEncoder                              #
#    - implementation of patch encoding as a layer #
# Methods:                                         #
#    - call(self, images)                          #
#    - get_config(self)                            #
####################################################
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

############################################
# mlp                                      #
#   - multilayer perceptron implementation #
############################################
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

################################################
# create_vit_encoder_decoder                   #
#   - creates a vision transformer model       #
#   - vision transformer for encoding,         #
#     CNNs for decoding/upsampling             #
################################################
def create_vit_encoder_decoder():
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs) # create patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) # encode patches

    # create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        x1 = encoded_patches
        # multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # mlp
        x3 = mlp(x2, hidden_units=transformer_units, dropout_rate=0.1)
        # skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # convolutional upscaling
    x = layers.Dense(projection_dim)(encoded_patches)
    x = layers.Reshape((image_size // patch_size, image_size // patch_size, projection_dim))(encoded_patches)

    # create multiple layers of conv
    for i in range(conv_layers):
        x = layers.Conv2DTranspose(
            filters = projection_dim // 2*(i+1),
            kernel_size = 3,
            strides = upscale_factor,
            padding = "same",
            activation = "relu"
        )(x)
    
    # final output activation
    outputs = layers.Conv2DTranspose(
        filters = 1,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        activation = "sigmoid"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary)
    
    return model

###################################################
# run_experiment                                  #
#   - compiles, trains, and evaluates given model #
###################################################
def run_experiment(model, x_train, y_train, x_test, y_test):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # compile model
    model.compile(
        optimizer=optimizer,
        loss = "mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"]
    )

    # save model checkpoint
    checkpoint_filepath = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_mean_squared_error",
        save_best_only=True,
        save_weights_only=True,
    )

    # fit model
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    # accuracy calculation is bugged currently: need to fix

    # evaluate accuracy
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

####################################################
# plot_history                                     #
#    - create a plot of model accuracy over epochs #
####################################################
def plot_history(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

##################################################################
# process_dataset                                                #
#   - Formats the prepared dataset to be suitable for the model, #
#     and splits it into a train test set.                       #
##################################################################
def process_dataset():
    # Get path to dataset
    dataset = "EarthRemoteSensingRapidResponse/Dataset/train_test"
    
    # Iterate over dataset and open each image with rasterio,
    # then split each image into X and Y and concat to a list
    X = []
    Y = []
    
    # iterate over dataset and append each image to list
    for image_file in os.listdir(dataset):
        image_path = os.path.join(dataset, image_file)
        
        # open image
        if os.path.isfile(image_path):
            with rasterio.open(image_path) as image:
                image_data = image.read()
                image_profile = image.profile
                
                # format input data
                image_data_t = np.array(image_data).transpose((1,2,0))
                X_split = np.array(image_data_t[:, :, :5])
                Y_split = np.array(image_data_t[:, :, 5:])
                
                # normalize X_split and Y_split - add 9999 (to Y_split only) then divide by max
                X_split_r = X_split / np.max(X_split)
                Y_split_r = (Y_split + 9999) / (np.max(Y_split + 9999))
                
                X.append(X_split_r)
                Y.append(Y_split_r)
    
    X = np.array(X)
    Y = np.array(Y) 
    # print(X.shape)
    # print(Y.shape)
    
    # split into 80/20 train test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # get normalization data for later
    # data_augmentation.layers[0].adapt(x_train)
    
    return (x_train, y_train), (x_test, y_test)

####################################################
# preprocess_image                                 #
#    - preprocess a single image                   #
####################################################
def preprocess_image(image_path):
    with rasterio.open(image_path) as image:
        image_data = image.read()

        image_profile = image.profile

        # format input data
        image_data_t = np.array(image_data).transpose((1,2,0))
        image_data_t = image_data_t[:, :, :5]
        
        # The model expects (# samples, 256, 256, 1), expand for batch dimension
        processed_image = np.expand_dims(image_data_t, axis=0)

    return processed_image

#################################################################
# errsr_model_prediction                                        #
#    - returns/displays methane prediction based on an s2 image #
#      and saves it as a kml                                    #
#################################################################
def errsr_model_prediction(image_path):
    checkpoint_filepath = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"

    model = create_vit_encoder_decoder()
    model.load_weights(checkpoint_filepath)
    processed_image = preprocess_image(image_path)
    
    methane_prediction = model.predict(processed_image)
    
    # print(methane_prediction[0])
    # print(np.array([methane_prediction[0]]).shape)
    
    # NOTE: the prediction seems to not be correct, likely due
    #       to errors with the model implementation (likely conv2dTranspose?).
    #       Could also be partly due to the small dataset size

    plt.imshow(methane_prediction[0])
    plt.title("Predicted Methane Plume Map")
    plt.show()
    
    # get profile of base input image
    with rasterio.open(image_path) as base_img:
        base_profile = base_img.profile
    
    # file path to save .tif to temporarily
    temp_path = "EarthRemoteSensingRapidResponse/Predictions/testprediction.tf"
        
    # save new .tif of prediction
    with rasterio.open(
        temp_path, 
        **{**base_profile, "count": 1},
        mode="w"
    ) as file:
        file.write(np.array(methane_prediction[0]).transpose(2,0,1))
        
    # get bounds of temp file
    with rasterio.open(temp_path) as t:
        # crs = t.crs
        bounds = t.bounds
        # print(crs)
        # print(bounds)
        
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        # print(west)
        # print(south)
        # print(east)
        # print(north)
    
    # write to kml and save
    kml = simplekml.Kml()
    ground_overlay = kml.newgroundoverlay(name=os.path.basename(temp_path))
    # ground_overlay.icon.href = temp_path
    ground_overlay.latlonbox.north = north
    ground_overlay.latlonbox.south = south
    ground_overlay.latlonbox.east = east
    ground_overlay.latlonbox.west = west
    
    # NOTE: this saves the data as a .kml, but its not visualized properly. 
    #       ideally we'd overlay the methane with the s2 image and save that,
    #       but we should probably just focus on visualizing the actual plume
    #       first for the sake of time
    
    kml.save("EarthRemoteSensingRapidResponse/Predictions/testprediction.kml")
        

########
# main #
########
def main():
    # process dataset
    (x_train, y_train), (x_test, y_test) = process_dataset()
    
    # create model and evaluate
    vit_classifier = create_vit_encoder_decoder()
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)

    # Input image into model and give prediction
    test_image_path = "EarthRemoteSensingRapidResponse/Dataset/validation/20230120T164611_20230120T164959_T15SYT.tif"
    errsr_model_prediction(test_image_path)

    # plot metrics
    plot_history(history, "loss")
    plot_history(history, "mean_squared_error")
    plot_history(history, "mean_absolute_error")
    
    return

if __name__ == "__main__":
    main()