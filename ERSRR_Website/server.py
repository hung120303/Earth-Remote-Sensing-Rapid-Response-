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

from rasterio.warp import calculate_default_transform, reproject, Resampling

import os

import mercantile
from PIL import Image
from io import BytesIO
import cv2

global prediction_data, prediction_profile

print("Server starting...")
checkpoint_dir = "../EarthRemoteSensingRapidResponse/tmp/checkpoint_new.weights.h5" # Weights
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

        pred_raw = model.predict(tile[None])

        # errsr_model_prediction code
        pred_norm = pred_raw["regression_output"][0, :, :, 0]
        pred_mask = pred_raw["mask_output"][0, :, :, 0]
        pred_norm_refined = pred_norm * pred_mask

        preds.append(pred_norm_refined)


    return preds

def stitch_predictions(preds, coords, shape, tile_size=256):
    H, W = shape
    output = np.zeros((H, W))

    for pred, (i, j) in zip(preds, coords):
        output[i:i+tile_size, j:j+tile_size] = pred

    return output

def save_prediction_rgba(output, profile):

    # Normalize to 0–1
    output = np.clip(output, 0, 1)

    # Create RGBA channels
    red   = (output * 255 * 2).clip(0,255).astype(np.uint8)
    green = np.zeros_like(red, dtype=np.uint8)
    blue  = np.zeros_like(red, dtype=np.uint8)

    # Alpha = transparency 
    threshold = 0.3
    alpha = np.where(output > threshold, output * 255, 0).astype(np.uint8)

    rgba = np.stack([red, green, blue, alpha], axis=0)

    rgba_3857, profile_3857 = reproject_to_3857(rgba, profile)

    global prediction_data, prediction_profile
    
    prediction_data = rgba_3857
    prediction_profile = profile_3857

    with rasterio.open(
        PRED_PATH,
        'w',
        driver='GTiff',
        height=profile_3857['height'],
        width=profile_3857['width'],
        count=4,
        dtype='uint8',
        crs=profile_3857['crs'],
        transform=profile_3857['transform']
    ) as dst:
        dst.write(rgba_3857)

def reproject_to_3857(data, profile):

    src_crs = profile['crs']
    dst_crs = 'EPSG:3857'

    bounds = rasterio.transform.array_bounds(
        profile['height'],
        profile['width'],
        profile['transform']
    )

    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        profile['width'],
        profile['height'],
        *bounds   
    )

    # prepare output array
    dst = np.zeros((data.shape[0], height, width), dtype=data.dtype)

    # reproject each band
    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=dst[i],
            src_transform=profile['transform'],
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # update profile
    new_profile = profile.copy()
    new_profile.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    return dst, new_profile

@app.route("/prediction")
def get_prediction():
    return send_file(PRED_PATH)

@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def get_tile(z, x, y):
    global prediction_data, prediction_profile

    if prediction_data is None:
        return "No prediction", 404

    bounds = mercantile.xy_bounds(x, y, z)

    transform = prediction_profile['transform']

    # convert bounds to pixel coordinates
    def world_to_pixel(x, y):
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        return col, row

    minx, miny, maxx, maxy = bounds

    col_min, row_max = world_to_pixel(minx, miny)
    col_max, row_min = world_to_pixel(maxx, maxy)

    # clamp to image bounds
    col_min = max(col_min, 0)
    row_min = max(row_min, 0)
    col_max = min(col_max, prediction_data.shape[2])
    row_max = min(row_max, prediction_data.shape[1])

    tile = prediction_data[:, row_min:row_max, col_min:col_max]

    if tile.size == 0:
        return "", 204

    # resize to 256x256
    tile = np.transpose(tile, (1,2,0))  # HWC
    tile_img = Image.fromarray(tile, mode='RGBA')
    tile_img = tile_img.resize((256,256), Image.BILINEAR)

    buf = BytesIO()
    tile_img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

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
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--ep', type=int, default=100)
parser.add_argument('--ps', type=int, default=8)
parser.add_argument('--nh', type=int, default=4)
parser.add_argument('--tl', type=int, default=8)
args = parser.parse_args()

MODEL_SETTING = "predict"

# data preparation
input_shape = (256, 256, 5)
output_shape = (256, 256, 1)

# hyperparameters
learning_rate = args.lr
weight_decay = args.wd
batch_size = args.bs
num_epochs = args.ep
image_size = 256
patch_size = args.ps
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = args.nh
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = args.tl

# File Directory paths
dataset_dir = "EarthRemoteSensingRapidResponse/Dataset/train_test"
#checkpoint_dir = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"
test_image = "EarthRemoteSensingRapidResponse/Dataset/validation/20240613T090559_20240613T091055_T34RET_EMIT_L2B_CH4PLM_001_20240612T135823_003244grid_1.tif"
cafo_csv = "EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv"

# Mask threshold config
PLUME_THRESHOLD_MODE = "percentile"   # "percentile" or "absolute"
PLUME_THRESHOLD_VALUE = 90            # top 10% methane pixels as plume

# Whether to gate regression output by predicted plume mask during inference
USE_MASK_GATING = True


####################################################
# class Patches                                    #
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
# class PatchEncoder                               #
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
############################################
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

####################################################
# transformer_block                                #
#   - implementation of a single transformer block #
####################################################
def transformer_block(x, n_heads, t_units):
    # multi-head attention layer
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=x.shape[-1], dropout=0.1)(x, x)
    # layer normalization and skip connection
    x = layers.LayerNormalization()(x + attn)
    # MLP
    mlp_output = mlp(x, hidden_units=t_units, dropout_rate=0.1)
    # layer normalization and skip connection
    x = layers.LayerNormalization()(x + mlp_output)
    return x

#############################################
# upsampling_block                          #
#############################################
def upsampling_block(x, filters):
    x = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x
    
    
def decoder_block(x, skip, filters):
    # upsample with transpose convolution
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding="same")(x)
    # add and concatenate skip connection
    skip = layers.Resizing(x.shape[1], x.shape[2])(skip)
    x = layers.Concatenate()([x, skip])
    # convolutional layers
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


################################################
# create_vit_encoder_decoder                   #
#   - plume-aware multi-task model             #
################################################
def create_vit_encoder_decoder():
    # ViT patches and encoding
    inputs = keras.Input(shape=input_shape)
    
    # # Skip connection to retain resolution
    # skip_conn = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    
    patches = Patches(patch_size)(inputs) # create patches
    x = PatchEncoder(num_patches, projection_dim)(patches) # encode patches
    
    # Transformer encoder
    skips = []
    for depth in range(transformer_layers):
        x = transformer_block(x, num_heads, transformer_units)
        if depth in [1, 4, 7]:
            skips.append(x)
    
    # for _ in range(transformer_layers):
    #     attn = layers.MultiHeadAttention(
    #         num_heads=num_heads,
    #         key_dim=projection_dim,
    #         dropout=0.1
    #     )(encoded_patches, encoded_patches)

    #     encoded_patches = layers.LayerNormalization()(encoded_patches + attn)
    #     mlp_output = mlp(encoded_patches, hidden_units=transformer_units, dropout_rate=0.1)
    #     encoded_patches = layers.LayerNormalization()(encoded_patches + mlp_output)

    # bridge
    h = image_size // patch_size
    x = layers.Reshape((h, h, x.shape[-1]))(x)
    for i, s in enumerate(skips):
        s = layers.Reshape((h, h, s.shape[-1]))(s)
        skips[i] = s
        
    # decoder layers
    x = decoder_block(x, skips[-1], projection_dim)
    x = decoder_block(x, skips[-2], projection_dim // 2)
    x = decoder_block(x, skips[-3], projection_dim // 4)

    # # Decoder
    # x = layers.Dense(projection_dim)(encoded_patches)
    # x = layers.Reshape((image_size // patch_size, image_size // patch_size, projection_dim))(x)

    # num_upsample_blocks = round(math.log(patch_size, 2))
    # for i in range(num_upsample_blocks):
    #     x = upsampling_block(x, (projection_dim // (2**i)))

    # skip_resized = layers.Conv2D(32, 3, padding="same", activation="relu")(skip_conn)
    # x = layers.Concatenate()([x, skip_resized])

    # x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    # x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    shared_features = x

    # Methane regression head (normalized to [0,1])
    regression_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="regression_output"
    )(shared_features)

    # Binary plume segmentation head
    mask_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="mask_output"
    )(shared_features)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "regression_output": regression_output,
            "mask_output": mask_output
        }
    )

    model.summary()
    return model


####################################################
# Regression losses / metrics                      #
####################################################
def masked_mse(y_true, y_pred):
    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)

    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)
    sq_err = ops.square(y_pred - y_true_safe) * valid_mask
    return ops.sum(sq_err) / (ops.sum(valid_mask) + 1e-6)


def masked_mae(y_true, y_pred):
    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)

    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)
    abs_err = ops.abs(y_pred - y_true_safe) * valid_mask
    return ops.sum(abs_err) / (ops.sum(valid_mask) + 1e-6)


def weighted_masked_mse(y_true, y_pred):
    """
    Gives higher weight to high methane pixels so plume regions matter more.
    """
    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)

    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)

    weights = 1.0 + 4.0 * y_true_safe   # tune later if needed
    weights = weights * valid_mask

    sq_err = ops.square(y_pred - y_true_safe) * weights
    return ops.sum(sq_err) / (ops.sum(weights) + 1e-6)


####################################################
# Segmentation losses / metrics                    #
####################################################
def masked_bce(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true = ops.expand_dims(y_true, axis=-1)

    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)

    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)

    bce = keras.losses.binary_crossentropy(y_true_safe, y_pred)

    if len(valid_mask.shape) == 4:
        valid_mask = ops.squeeze(valid_mask, axis=-1)

    bce = bce * valid_mask
    return ops.sum(bce) / (ops.sum(valid_mask) + 1e-6)


def masked_dice_loss(y_true, y_pred, smooth=1e-6):
    valid_mask = ops.logical_not(ops.isnan(y_true))
    valid_mask = ops.cast(valid_mask, y_pred.dtype)

    y_true_safe = ops.where(ops.isnan(y_true), ops.zeros_like(y_true), y_true)

    y_true_flat = ops.reshape(y_true_safe * valid_mask, (-1,))
    y_pred_flat = ops.reshape(y_pred * valid_mask, (-1,))

    intersection = ops.sum(y_true_flat * y_pred_flat)
    denom = ops.sum(y_true_flat) + ops.sum(y_pred_flat)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice


def plume_segmentation_loss(y_true, y_pred):
    return 0.5 * masked_bce(y_true, y_pred) + 0.5 * masked_dice_loss(y_true, y_pred)


####################################################
# normalize_emit_local                             #
####################################################
def normalize_emit_local(emit, nodata_value=-9999, use_log=True, p_low=2, p_high=98):
    emit = emit.astype(np.float32)
    valid_mask = (emit != nodata_value)

    if not np.any(valid_mask):
        return None, None

    emit_proc = emit.copy()

    if use_log:
        emit_proc[valid_mask] = np.log1p(np.maximum(emit_proc[valid_mask], 0.0))

    valid_vals = emit_proc[valid_mask]

    low = np.percentile(valid_vals, p_low)
    high = np.percentile(valid_vals, p_high)

    if np.isclose(high, low):
        norm = np.zeros_like(emit_proc, dtype=np.float32)
    else:
        norm = (emit_proc - low) / (high - low + 1e-6)

    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    norm[~valid_mask] = np.nan

    stats = {
        "low": float(low),
        "high": float(high),
        "use_log": use_log,
        "nodata_value": nodata_value,
        "p_low": p_low,
        "p_high": p_high,
    }

    return norm, stats


####################################################
# denormalize_emit_local                           #
####################################################
def denormalize_emit_local(pred_norm, stats):
    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    low = stats["low"]
    high = stats["high"]
    use_log = stats["use_log"]

    pred = pred_norm * (high - low + 1e-6) + low

    if use_log:
        pred = np.expm1(pred)

    pred = np.maximum(pred, 0.0).astype(np.float32)
    return pred


####################################################
# create_plume_mask                                #
#   - binary target for plume segmentation         #
####################################################
def create_plume_mask(emit, nodata_value=-9999, threshold_mode="percentile", threshold_value=90):
    """
    emit: shape (H, W, 1) or (H, W)
    returns:
        plume_mask: float32 array with:
            1.0 = plume
            0.0 = non-plume
            NaN = nodata
    """
    emit = emit.astype(np.float32)

    if emit.ndim == 3:
        emit_2d = emit[:, :, 0]
    else:
        emit_2d = emit

    valid_mask = (emit_2d != nodata_value) & np.isfinite(emit_2d)

    plume_mask = np.full_like(emit_2d, np.nan, dtype=np.float32)

    if not np.any(valid_mask):
        return plume_mask[..., None]

    valid_vals = emit_2d[valid_mask]
    valid_vals_log = np.log1p(np.maximum(valid_vals, 0.0))

    if threshold_mode == "percentile":
        thresh = np.percentile(valid_vals_log, threshold_value)
    elif threshold_mode == "absolute":
        thresh = threshold_value
    else:
        raise ValueError("threshold_mode must be 'percentile' or 'absolute'")

    emit_log = np.zeros_like(emit_2d, dtype=np.float32)
    emit_log[valid_mask] = np.log1p(np.maximum(emit_2d[valid_mask], 0.0))

    plume_binary = (emit_log >= thresh).astype(np.float32)
    plume_mask[valid_mask] = plume_binary[valid_mask]

    return plume_mask[..., None]


##################################################################
# process_dataset                                                #
##################################################################
def process_dataset():
    X = []
    Y_reg = []
    Y_mask = []
    target_stats = []

    for root, subfolders, filenames in os.walk(dataset_dir):
        for image_file in filenames:
            image_path = os.path.join(root, image_file)
            if os.path.isfile(image_path):
                with rasterio.open(image_path) as image:
                    image_data = image.read().transpose((1, 2, 0)).astype(np.float32)

                    # Input bands
                    X_split = image_data[:, :, :5].astype(np.float32)

                    # Methane target
                    emit = image_data[:, :, 5:6].astype(np.float32)

                    X_split = np.nan_to_num(X_split, nan=0.0, posinf=0.0, neginf=0.0)

                    # Regression target
                    Y_split, stats = normalize_emit_local(
                        emit,
                        nodata_value=-9999,
                        use_log=True,
                        p_low=2,
                        p_high=98
                    )

                    # Segmentation target
                    mask_split = create_plume_mask(
                        emit,
                        nodata_value=-9999,
                        threshold_mode=PLUME_THRESHOLD_MODE,
                        threshold_value=PLUME_THRESHOLD_VALUE
                    )

                    if Y_split is None or stats is None:
                        continue

                    X.append(X_split)
                    Y_reg.append(Y_split)
                    Y_mask.append(mask_split)

                    target_stats.append({
                        "image_path": image_path,
                        **stats
                    })

    X = np.array(X, dtype=np.float32)
    Y_reg = np.array(Y_reg, dtype=np.float32)
    Y_mask = np.array(Y_mask, dtype=np.float32)

    finite_x = np.isfinite(X)
    x_max = np.max(np.abs(X[finite_x]))
    if x_max < 1e-8:
        x_max = 1.0

    X = X / (x_max + 1e-6)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    np.save("EarthRemoteSensingRapidResponse/tmp/x_max.npy", x_max)

    with open("EarthRemoteSensingRapidResponse/tmp/target_stats.json", "w") as f:
        json.dump(target_stats, f, indent=2)

    ratio_split = 0.2
    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)

    split_point = int(len(indices) * (1 - ratio_split))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    x_train = X[train_indices]
    x_test = X[test_indices]

    y_reg_train = Y_reg[train_indices]
    y_mask_train = Y_mask[train_indices]

    y_reg_test = Y_reg[test_indices]
    y_mask_test = Y_mask[test_indices]

    print(f"x_train: {x_train.shape}")
    print(f"y_reg_train: {y_reg_train.shape}, y_mask_train: {y_mask_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_reg_test: {y_reg_test.shape}, y_mask_test: {y_mask_test.shape}")

    valid_train = y_reg_train[np.isfinite(y_reg_train)]
    print("Valid y_reg_train stats:", valid_train.min(), valid_train.max(), valid_train.mean())

    print("NaN check -> x_train:", np.isnan(x_train).any(),
          "y_reg_train valid count:", np.isfinite(y_reg_train).sum(),
          "y_mask_train valid count:", np.isfinite(y_mask_train).sum(),
          "x_test:", np.isnan(x_test).any(),
          "y_reg_test valid count:", np.isfinite(y_reg_test).sum(),
          "y_mask_test valid count:", np.isfinite(y_mask_test).sum())

    return (
        x_train,
        {
            "regression_output": y_reg_train,
            "mask_output": y_mask_train
        }
    ), (
        x_test,
        {
            "regression_output": y_reg_test,
            "mask_output": y_mask_test
        }
    )


####################################################
# preprocess_image                                 #
####################################################
def preprocess_image(image_path):
    x_max = np.load("EarthRemoteSensingRapidResponse/tmp/x_max.npy")

    with rasterio.open(image_path) as image:
        image_data = image.read().transpose(1,2,0)[:, :, :5].astype(np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        image_data = image_data / (x_max + 1e-6)
    return image_data


###################################################
# run_experiment                                  #
###################################################
def run_experiment(model, x_train, y_train, x_test, y_test):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "regression_output": weighted_masked_mse,
            "mask_output": plume_segmentation_loss,
        },
        loss_weights={
            "regression_output": 1.0,
            "mask_output": 1.0,
        },
        metrics={
            "regression_output": [masked_mae, masked_mse],
            "mask_output": [masked_bce],
        }
    )

    checkpoint_filepath = checkpoint_dir
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_regression_output_masked_mse",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    eval_results = model.evaluate(x_test, y_test, return_dict=True)

    print("\n===== TEST METRICS =====")
    for key, val in eval_results.items():
        print(f"{key}: {val}")

    return history


####################################################
# plot_history                                     #
####################################################
def plot_history(history, item):
    if item not in history.history:
        print(f"Skipping plot: '{item}' not found in history.")
        print("Available history keys:", list(history.history.keys()))
        return

    val_item = "val_" + item

    plt.figure(figsize=(8,5))
    plt.plot(history.history[item], label=item)

    if val_item in history.history:
        plt.plot(history.history[val_item], label=val_item)

    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

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




    save_prediction_rgba(output, profile)
    
        
    # # save new .tif of prediction
    # with rasterio.open(
    #     temp_path, 
    #     mode="w",
    #     driver='GTiff',
    #     height=output.shape[0],
    #     width=output.shape[1],
    #     count=1,
    #     dtype='float32',
    #     crs=profile['crs'],
    #     transform=profile['transform']
    # ) as file:
    #     file.write(np.expand_dims(output, axis=0))
        
    # # get bounds of temp file
    # with rasterio.open(temp_path) as t:
    #     # crs = t.crs
    #     bounds = t.bounds
    #     # print(crs)
    #     # print(bounds)
        
    #     west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    #     # print(west)
    #     # print(south)
    #     # print(east)
    #     # print(north)
    
    # # write to kml and save
    # kml = simplekml.Kml()
    # ground_overlay = kml.newgroundoverlay(name=os.path.basename(temp_path))
    # # ground_overlay.icon.href = temp_path
    # ground_overlay.latlonbox.north = north
    # ground_overlay.latlonbox.south = south
    # ground_overlay.latlonbox.east = east
    # ground_overlay.latlonbox.west = west
    
    # kml.save(temp_path)

def errsr_model_prediction_wide(image_path):

    # load image
    with rasterio.open(image_path) as src:
        img = src.read().transpose(1,2,0)   
        profile = src.profile

    H, W, C = img.shape

    # resize to model input
    img_resized = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)

    # normalize
    img_resized = img_resized.astype(np.float32) / 10000.0

    # run model
    pred = model.predict(img_resized[None], verbose=0)[0,:,:,0]

    # resize
    pred_full = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)

    # rgb display
    pred_full = np.clip(pred_full, 0, 1)

    threshold = 0.3

    red = (pred_full * 255).astype(np.uint8)
    green = np.zeros_like(red)
    blue = np.zeros_like(red)

    alpha = np.where(pred_full > threshold, pred_full * 255, 0).astype(np.uint8)

    rgba = np.stack([red, green, blue, alpha], axis=0)

    rgba_3857, profile_3857 = reproject_to_3857(rgba, profile)

    global prediction_data, prediction_profile
    
    prediction_data = rgba_3857
    prediction_profile = profile_3857

    # save tif file
    os.makedirs("Predictions", exist_ok=True)

    with rasterio.open(
        PRED_PATH,
        'w',
        driver='GTiff',
        height=H,
        width=W,
        count=4,
        dtype='uint8',
        crs=profile['crs'],
        transform=profile['transform']
    ) as dst:
        dst.write(rgba)

    

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