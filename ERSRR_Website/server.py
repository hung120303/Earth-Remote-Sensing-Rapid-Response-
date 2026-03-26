import ee
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

import keras
from keras import layers
from keras import ops

import requests

import rasterio
import simplekml

import os

print("Server starting...")

checkpoint_dir = "../EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5" # Weights
temp_path = "Predictions/testprediction.kml"
PRED_PATH = "Predictions/testprediction.tif"

ee.Authenticate() 
ee.Initialize(project='ersrr-475700') # Ensure this project ID is correct and you have access

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

TILE_SIZE = 256

def load_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read()  # (bands, H, W)
        profile = src.profile

    img = np.transpose(img, (1, 2, 0))  # (H, W, bands)
    return img, profile

def pad_image(img, tile_size=256):
    H, W, C = img.shape

    new_H = ((H // tile_size) + 1) * tile_size
    new_W = ((W // tile_size) + 1) * tile_size

    padded = np.zeros((new_H, new_W, C))
    padded[:H, :W, :] = img

    return padded, H, W

def create_tiles(img, tile_size=256):
    H, W, C = img.shape
    tiles = []
    coords = []

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):

            tile = img[i:i+tile_size, j:j+tile_size]

            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            tiles.append(tile)
            coords.append((i, j))

    return tiles, coords

def predict_tiles(model, tiles):
    preds = []

    for tile in tiles:
        tile = tile / (np.max(tile) + 1e-6)  # normalize

        pred = model.predict(tile[None])[0, :, :, 0]
        preds.append(pred)

    return preds

def stitch_predictions(preds, coords, shape, tile_size=256):
    H, W = shape
    output = np.zeros((H, W))

    for pred, (i, j) in zip(preds, coords):
        output[i:i+tile_size, j:j+tile_size] = pred

    return output
@app.route("/prediction")
def get_prediction():
    return send_file(PRED_PATH)

@app.route("/sentinel")
def sentinel():

    minx = float(request.args.get("minx"))
    miny = float(request.args.get("miny"))
    maxx = float(request.args.get("maxx"))
    maxy = float(request.args.get("maxy"))

    roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(roi)
        .filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'LESS_THAN', 5)
        .filterDate("2022-01-01","2022-12-31")
        .first()
        .select(["B4","B3","B2", "B11", "B12"])
    )

    download_url = image.getDownloadURL({
        'region': roi.getInfo(),
        'scale': 20,
        'format': 'GEO_TIFF'
    })

    s2_path = "temp_s2.tif"

    response = requests.get(download_url)
    with open(s2_path, "wb") as f:
        f.write(response.content)

    errsr_model_prediction(s2_path)

    vis = {"min":0,"max":3000,"bands":["B4","B3","B2"]}
    map_id = image.getMapId(vis)

    return jsonify({
        "tile_url": map_id["tile_fetcher"].url_format
    })

''' 
#########################################
    Model Code 
#########################################
'''

# accept arguments for hyperparameter testing
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="pred")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--ep', type=int, default=100)
parser.add_argument('--ps', type=int, default=8)
parser.add_argument('--nh', type=int, default=4)
parser.add_argument('--tl', type=int, default=8)
args = parser.parse_args()

# data preparation
input_shape = (256, 256, 5)
output_shape = (256, 256, 1)

# hyperparameters
learning_rate = args.lr
weight_decay = args.wd
batch_size = args.bs
num_epochs = args.ep
image_size = 256 # input is resized to (image_size x image_size) 
patch_size = args.ps  # size of patches to extract from input
num_patches = (image_size // patch_size) ** 2
projection_dim = 64 # set to 64 for small datasets, otherwise 768 or 1024
num_heads = args.nh
# upscale_factor = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
] # size of transformer layers
transformer_layers = args.tl
# conv_layers = 4
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

#############################################
# upsampling_block                          #
#   - convolutional upsampling block        #
#############################################
def upsampling_block(x,filters):
    x = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
    x = layers.Conv2D(filters, 3, padding="same",activation="relu")(x)
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

        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1
                                         )(encoded_patches, encoded_patches)
        encoded_patches = layers.LayerNormalization()(encoded_patches+attn)
        mlp_output = mlp(encoded_patches, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.LayerNormalization()(encoded_patches + mlp_output)

    # convolutional upscaling
    x = layers.Dense(projection_dim)(encoded_patches)
    x = layers.Reshape((image_size // patch_size, image_size // patch_size, projection_dim))(x)

    # upsampling blocks
    num_upsample_blocks = round(math.log(patch_size, 2))
    for i in range(num_upsample_blocks):
        x = upsampling_block(x, (projection_dim // (2**i)))

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = keras.ops.sum(y_true * y_pred, axis=[1, 2, 3])
    union = keras.ops.sum(y_true, axis=[1, 2, 3]) + keras.ops.sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return ops.mean(dice)

bce = keras.losses.BinaryCrossentropy()

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def hybrid_loss(y_true, y_pred):

    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

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
        loss = hybrid_loss,
        metrics=["mean_absolute_error", dice_coefficient]
    )

    # save model checkpoint
    checkpoint_filepath = checkpoint_dir
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_mean_squared_error",
        mode="min",
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
    dataset = dataset_dir
    
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
                image_data = image.read().transpose((1,2,0))
                
                # format input data
                X_split = image_data[:, :, :5]
                Y_split = image_data[:, :, 5:6]
                
                X.append(X_split)
                Y.append(Y_split)
    
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)

    x_Max = X.max()
    X = X / (x_Max + 1e-6) # normalize input data

    y_Min = Y.min()
    y_Max = Y.max()
    Y = (Y - y_Min) / (y_Max - y_Min + 1e-6) # normalize output data
    
    
    # split into 80/20 train test
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    ratio_split = 0.2 
    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_point = int(len(indices) * (1 - ratio_split))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    x_train = X[train_indices]
    y_train = Y[train_indices]
    x_test = X[test_indices]
    y_test = Y[test_indices]
    
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
        image_data = image.read().transpose(1,2,0)[0:, :, :5]
        image_data = image_data / (np.max(image_data) + 1e-6)  # normalize input
        
        # The model expects (# samples, 256, 256, 1), expand for batch dimension
        # processed_image = np.expand_dims(image_data_t, axis=0)

    return image_data

def errsr_model_prediction(image_path):
    checkpoint_filepath = checkpoint_dir

    # model = create_vit_encoder_decoder()
    # model.load_weights(checkpoint_filepath)

    # Load image
    img, profile = load_image(image_path)

    # Pad image
    padded_img, orig_H, orig_W = pad_image(img, TILE_SIZE)

    # Create tiles
    tiles, coords = create_tiles(padded_img, TILE_SIZE)

    # Predict
    preds = predict_tiles(model, tiles)

    # Stitch
    output = stitch_predictions(
        preds,
        coords,
        padded_img.shape[:2],
        TILE_SIZE
    )

    # Remove padding
    output = output[:orig_H, :orig_W]


    # file path to save .tif to temporarily
    # temp_path = "EarthRemoteSensingRapidResponse/ERSRR_Website/Predictions/testprediction.kml"
        
        
    # save new .tif of prediction
    with rasterio.open(
        temp_path, 
        mode="w",
        driver='GTiff',
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype='float32',
        crs=profile['crs'],
        transform=profile['transform']
    ) as file:
        file.write(np.expand_dims(output, axis=0))
        
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
    
    kml.save(temp_path)

'''
##############################
End model code
##############################
'''

if __name__ == "__main__":
    checkpoint_filepath = checkpoint_dir
    model = create_vit_encoder_decoder()
    model.load_weights(checkpoint_filepath)
    app.run(host="localhost", port=5000, debug=True)