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
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys

import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import json

import keras
from keras import layers
from keras import ops
# from sklearn.model_selection import train_test_split 
import simplekml
import rasterio
from rasterio.plot import show

from Plume_Classifier import evaluate_model

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

MODEL_SETTING = "train"

# data preparation
input_shape = (256, 256, 5)
output_shape = (256, 256, 1)

# hyperparameters
learning_rate = 0.0001
weight_decay = 0.1
batch_size = 4
num_epochs = 20
image_size = 256 # input is resized to (image_size x image_size) 
patch_size = 8  # size of patches to extract from input
num_patches = (image_size // patch_size) ** 2
projection_dim = 64 # set to 64 for small datasets, otherwise 768 or 1024
num_heads = 4
# upscale_factor = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
] # size of transformer layers
transformer_layers = 8
# conv_layers = 4
mlp_head_units = [
    2048,
    1024,
] # size of dense layers for final classifier (temp: need to adjust)

# File Directory paths
dataset_dir = "EarthRemoteSensingRapidResponse/Dataset/train_test_s2_100"
checkpoint_dir = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"
test_image = "EarthRemoteSensingRapidResponse/Dataset/validation/20240613T090559_20240613T091055_T34RET_EMIT_L2B_CH4PLM_001_20240612T135823_003244grid_1.tif"
cafo_csv = "EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv"

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

    # Skip connection to retain resolution
    skip_conn = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)

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

    skip_resized = layers.Conv2D(32, 3, padding="same", activation="relu")(skip_conn)
    x = layers.Concatenate()([x, skip_resized])

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(1, 1, activation="relu")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    return model

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
        loss = masked_mse,
        metrics=[masked_mae, masked_mse]
    )

    # save model checkpoint
    checkpoint_filepath = checkpoint_dir
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_masked_mse",
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
    loss, mae, mse = model.evaluate(x_test, y_test)
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")

    return history

####################################################
# plot_history                                     #
#    - create a plot of model accuracy over epochs #
####################################################
def plot_history(history, item):
    if item not in history.history:
        print(f"Skipping plot: '{item}' not found in history.")
        print("Available history keys:", list(history.history.keys()))
        return

    val_item = "val_" + item

    plt.plot(history.history[item], label=item)

    if val_item in history.history:
        plt.plot(history.history[val_item], label=val_item)

    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

####################################################
# normalize_emit_local                             #
#   - local percentile normalization for EMIT      #
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
#   - invert local percentile normalization        #
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

##################################################################
# process_dataset                                                #
#   - Formats the prepared dataset to be suitable for the model, #
#     and splits it into a train test set.                       #
##################################################################
def process_dataset():
    X = []
    Y = []
    target_stats = []

    for root, subfolders, filenames in os.walk(dataset_dir):
        for image_file in filenames:
            image_path = os.path.join(root, image_file)
            if os.path.isfile(image_path):
                with rasterio.open(image_path) as image:
                    image_data = image.read().transpose((1, 2, 0)).astype(np.float32)

                    # format input data
                    X_split = image_data[:, :, :5].astype(np.float32)
                    emit = image_data[:, :, 5:6].astype(np.float32)

                    X_split = np.nan_to_num(X_split, nan=0.0, posinf=0.0, neginf=0.0)

                    Y_split, stats = normalize_emit_local(
                        emit,
                        nodata_value=-9999,
                        use_log=True,
                        p_low=2,
                        p_high=98
                    )
                    X.append(X_split)
                    Y.append(Y_split)
                    target_stats.append({
                        "image_path": image_path,
                        **stats
                    })

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

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
    y_train = Y[train_indices]
    x_test = X[test_indices]
    y_test = Y[test_indices]

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    valid_train = y_train[np.isfinite(y_train)]
    print("Valid y_train stats:", valid_train.min(), valid_train.max(), valid_train.mean())

    print("NaN check -> x_train:", np.isnan(x_train).any(),
          "y_train valid count:", np.isfinite(y_train).sum(),
          "x_test:", np.isnan(x_test).any(),
          "y_test valid count:", np.isfinite(y_test).sum())

    return (x_train, y_train), (x_test, y_test)

####################################################
# preprocess_image                                 #
#    - preprocess a single image                   #
####################################################
def preprocess_image(image_path):
    x_max = np.load("EarthRemoteSensingRapidResponse/tmp/x_max.npy")

    with rasterio.open(image_path) as image:
        image_data = image.read().transpose(1,2,0)[0:, :, :5].astype(np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        image_data = image_data / (x_max + 1e-6)  # normalize input
        
        # The model expects (# samples, 256, 256, 1), expand for batch dimension
        # processed_image = np.expand_dims(image_data_t, axis=0)
    return image_data

#################################################################
# errsr_model_prediction                                        #
#    - returns/displays methane prediction based on an s2 image #
#      and saves it as a kml                                    #
#################################################################
def errsr_model_prediction(image_path):
    checkpoint_filepath = checkpoint_dir

    model = create_vit_encoder_decoder()
    model.load_weights(checkpoint_filepath)

    # Predict normalized plume map
    pred_norm = model.predict(preprocess_image(image_path)[None], verbose=0)[0, :, :, 0]
    pred_norm = np.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    with rasterio.open(image_path) as base_img:
        emit = base_img.read(6).astype(np.float32)
        base_profile = base_img.profile
        base_transform = base_img.transform

        # rasterio is 1-indexed
        b = base_img.read(1)
        g = base_img.read(2)
        r = base_img.read(3)
        b11 = base_img.read(4)
        b12 = base_img.read(5)

    emit_mask = (emit == -9999)
    valid_mask = ~emit_mask

    # Ground truth normalized using SAME training normalization recipe
    emit_norm, stats = normalize_emit_local(
        emit[..., None],
        nodata_value=-9999,
        use_log=True,
        p_low=2,
        p_high=98
    )

    if emit_norm is None:
        raise ValueError(f"No valid EMIT values found in {image_path}")

    emit_norm = emit_norm[:, :, 0]

    # De-normalize prediction for methane-like values
    pred_denorm = denormalize_emit_local(pred_norm, stats)

    # RAW prediction = honest full prediction
    pred_norm_raw = pred_norm.copy()
    pred_denorm_raw = pred_denorm.copy()

    # MASKED prediction = only for side-by-side comparison
    pred_norm_masked = pred_norm.copy()
    pred_norm_masked[emit_mask] = np.nan

    pred_denorm_masked = pred_denorm.copy()
    pred_denorm_masked[emit_mask] = np.nan

    print("plume accuracy stats:", evaluate_model(pred_denorm_masked, cafo_csv, image_path))
    print("Pred Norm min:", np.nanmin(pred_norm_raw))
    print("Pred Norm max:", np.nanmax(pred_norm_raw))
    print("Pred Norm mean:", np.nanmean(pred_norm_raw))
    print("Pred Norm std:", np.nanstd(pred_norm_raw))

    print("Pred Denorm min:", np.nanmin(pred_denorm_raw))
    print("Pred Denorm max:", np.nanmax(pred_denorm_raw))
    print("Pred Denorm mean:", np.nanmean(pred_denorm_raw))
    print("Pred Denorm std:", np.nanstd(pred_denorm_raw))

    # normalize rgb values to 0-255
    rgb = np.dstack((
        (((r - r.min()) / (r.max() - r.min() + 1e-6)) * 255),
        (((g - g.min()) / (g.max() - g.min() + 1e-6)) * 255),
        (((b - b.min()) / (b.max() - b.min() + 1e-6)) * 255))
    ).astype('uint8')
    rgb = np.transpose(rgb, (2, 0, 1))

    # safe log plotting
    pred_denorm_plot = np.where(np.isfinite(pred_denorm_raw), np.maximum(pred_denorm_raw, 0.0), np.nan)

    # create and save subplot figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), layout="constrained")

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    show(pred_norm_raw, ax=axes[0, 0], cmap="cividis")
    axes[0, 0].set_title("Predicted Normalized Plume (RAW)")

    show(pred_norm_masked, ax=axes[0, 1], cmap="cividis")
    axes[0, 1].set_title("Predicted Normalized Plume (Masked)")

    show(emit_norm, ax=axes[0, 2], cmap="cividis")
    axes[0, 2].set_title("Ground Truth Normalized EMIT")

    show(np.log1p(pred_denorm_plot), ax=axes[1, 0], cmap="cividis")
    axes[1, 0].set_title("Predicted Methane Plume (log scale)")

    show(rgb, ax=axes[1, 1])
    axes[1, 1].set_title('RGB S2 Input (Bands 2, 3, 4)')

    show(b12, ax=axes[1, 2])
    axes[1, 2].set_title('SWIR-2 S2 Input (Band 12)')

    fig.savefig("EarthRemoteSensingRapidResponse/Predictions/testprediction.png")
    plt.show()

    # save RAW prediction tif (not ground-truth masked)
    temp_path = "EarthRemoteSensingRapidResponse/Predictions/testprediction.tif"

    with rasterio.open(
        temp_path,
        **{**base_profile, "count": 1},
        mode="w"
    ) as file:
        file.write(np.expand_dims(pred_denorm_raw, axis=0))

    with rasterio.open(temp_path) as t:
        bounds = t.bounds
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    kml = simplekml.Kml()
    ground_overlay = kml.newgroundoverlay(name=os.path.basename(temp_path))
    ground_overlay.latlonbox.north = north
    ground_overlay.latlonbox.south = south
    ground_overlay.latlonbox.east = east
    ground_overlay.latlonbox.west = west

    kml.save("EarthRemoteSensingRapidResponse/Predictions/testprediction.kml")

########
# main #
########
def train_model():    
    # process dataset
    (x_train, y_train), (x_test, y_test) = process_dataset()
        
    # create model and evaluate
    vit_classifier = create_vit_encoder_decoder()
    history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)

    # Input image into model and give prediction
    
    # example images used for validation:
    # 20230410T172859_20230410T174507_T13RFQ.tif
    # 20240613T090559_20240613T091055_T34RET_EMIT_L2B_CH4PLM_001_20240612T135823_003244grid_1.tif
    # 20240422T105621_20240422T110339_T30SVJ_EMIT_L2B_CH4PLM_001_20240416T131651_003132grid_1.tif
    # 20231029T072021_20231029T072130_T39SVV_EMIT_L2B_CH4PLM_001_20231027T061434_001888grid_1.tif
    # 20240403T170851_20240403T171803_T14RMS_EMIT_L2B_CH4PLM_001_20240322T214509_002926grid_1.tif
    
    
    errsr_model_prediction(test_image)

    # plot metrics
    plot_history(history, "loss")
    plot_history(history, "masked_mse")
    plot_history(history, "masked_mae")

if __name__ == "__main__":
    if(MODEL_SETTING == "train"):
        train_model()
    elif(MODEL_SETTING == "predict"):
        errsr_model_prediction(test_image)