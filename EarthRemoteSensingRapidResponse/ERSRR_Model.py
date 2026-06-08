#################################################################################################################################
# ERSRR_Model.py                                                                                                                #
#   - This program file handles training the model, evaluating its accuracy and prediction tasks using the generated model.     #
#     Modified into a plume-aware multi-task model:                                                                             #
#       1) plume segmentation branch (binary mask)                                                                              #
#       2) methane regression branch (continuous methane intensity)                                                             #
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
checkpoint_dir = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"
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


#################################################################
# errsr_model_prediction                                        #
#################################################################
def errsr_model_prediction(image_path):
    checkpoint_filepath = checkpoint_dir

    model = create_vit_encoder_decoder()

    if not os.path.exists(checkpoint_filepath):
        print(f"No checkpoint found at {checkpoint_filepath}. Train first with MODEL_SETTING='train'.")
        return
    try:
        model.load_weights(checkpoint_filepath)
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("The checkpoint format is incompatible. Retrain the model first.")
        return

    preds = model.predict(preprocess_image(image_path)[None], verbose=0)

    pred_norm = preds["regression_output"][0, :, :, 0]
    pred_mask = preds["mask_output"][0, :, :, 0]

    pred_norm = np.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
    pred_mask = np.nan_to_num(pred_mask, nan=0.0, posinf=0.0, neginf=0.0)

    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    pred_mask = np.clip(pred_mask, 0.0, 1.0)

    # Optional refinement: suppress methane where mask branch says "not plume"
    if USE_MASK_GATING:
        pred_norm_refined = pred_norm * pred_mask
    else:
        pred_norm_refined = pred_norm.copy()

    with rasterio.open(image_path) as base_img:
        emit = base_img.read(6).astype(np.float32)
        base_profile = base_img.profile
        base_transform = base_img.transform

        b = base_img.read(1)
        g = base_img.read(2)
        r = base_img.read(3)
        b11 = base_img.read(4)
        b12 = base_img.read(5)

    emit_mask = (emit == -9999)

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

    true_mask = create_plume_mask(
        emit[..., None],
        nodata_value=-9999,
        threshold_mode=PLUME_THRESHOLD_MODE,
        threshold_value=PLUME_THRESHOLD_VALUE
    )[:, :, 0]

    pred_denorm = denormalize_emit_local(pred_norm_refined, stats)

    # RAW / masked versions
    pred_norm_raw = pred_norm_refined.copy()
    pred_denorm_raw = pred_denorm.copy()

    pred_norm_masked = pred_norm_refined.copy()
    pred_norm_masked[emit_mask] = np.nan

    pred_mask_plot = pred_mask.copy()
    pred_mask_plot[emit_mask] = np.nan

    pred_denorm_masked = pred_denorm.copy()
    pred_denorm_masked[emit_mask] = np.nan

    print("plume accuracy stats:", evaluate_model(pred_denorm_masked, cafo_csv, image_path))
    print("Pred Norm min:", np.nanmin(pred_norm_raw))
    print("Pred Norm max:", np.nanmax(pred_norm_raw))
    print("Pred Norm mean:", np.nanmean(pred_norm_raw))
    print("Pred Norm std:", np.nanstd(pred_norm_raw))

    print("Pred Mask min:", np.nanmin(pred_mask))
    print("Pred Mask max:", np.nanmax(pred_mask))
    print("Pred Mask mean:", np.nanmean(pred_mask))
    print("Pred Mask std:", np.nanstd(pred_mask))

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

    pred_denorm_plot = np.where(np.isfinite(pred_denorm_raw), np.maximum(pred_denorm_raw, 0.0), np.nan)

    # 2x3 visualization
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), layout="constrained")

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    show(pred_norm_raw, ax=axes[0, 0], cmap="cividis")
    axes[0, 0].set_title("Predicted Normalized Methane")

    show(pred_mask_plot, ax=axes[0, 1], cmap="magma")
    axes[0, 1].set_title("Predicted Plume Mask")

    show(emit_norm, ax=axes[0, 2], cmap="cividis")
    axes[0, 2].set_title("Ground Truth Normalized EMIT")

    show(true_mask, ax=axes[1, 0], cmap="magma")
    axes[1, 0].set_title("Ground Truth Plume Mask")

    show(rgb, ax=axes[1, 1])
    axes[1, 1].set_title("RGB S2 Input (Bands 2, 3, 4)")

    show(b12, ax=axes[1, 2], cmap="gray")
    axes[1, 2].set_title("SWIR-2 S2 Input (Band 12)")

    fig.savefig("EarthRemoteSensingRapidResponse/Predictions/testprediction.png")
    plt.show()

    # save prediction tif
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
    (x_train, y_train), (x_test, y_test) = process_dataset()
        
    vit_model = create_vit_encoder_decoder()
    history = run_experiment(vit_model, x_train, y_train, x_test, y_test)

    errsr_model_prediction(test_image)

    # useful history plots
    plot_history(history, "loss")
    plot_history(history, "regression_output_loss")
    plot_history(history, "mask_output_loss")
    plot_history(history, "regression_output_masked_mse")
    plot_history(history, "regression_output_masked_mae")
    plot_history(history, "mask_output_masked_bce")


if __name__ == "__main__":
    if MODEL_SETTING == "train":
        train_model()
    elif MODEL_SETTING == "predict":
        errsr_model_prediction(test_image)