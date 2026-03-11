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
upscale_factor = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
] # size of transformer layers
transformer_layers = 8
conv_layers = 4
mlp_head_units = [
    2048,
    1024,
] # size of dense layers for final classifier (temp: need to adjust)

# File Directory paths
dataset_dir = "EarthRemoteSensingRapidResponse/Data Collection/train_test_2_13_good_data"
checkpoint_dir = "EarthRemoteSensingRapidResponse/tmp/checkpoint_new_data.weights.h5"
test_image = "EarthRemoteSensingRapidResponse/Dataset/validation/20241010T102941_20241010T103402_T31SES_EMIT_L2B_CH4PLM_001_20241006T090738_003685grid_1.tif"

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

    num_upsample_blocks = round(math.log(patch_size, 2))
    
    for i in range(num_upsample_blocks):
        x = upsampling_block(x, (projection_dim // (2**i)))

    # x = upsampling_block(x, 64)
    # x = upsampling_block(x, 32) 
    # x = upsampling_block(x, 16)
    # x = upsampling_block(x, 8)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary)
    
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
        image_data = image.read().transpose(1,2,0)[0:, :, :5]
        image_data = image_data / (np.max(image_data) + 1e-6)  # normalize input
        
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
    #processed_image = preprocess_image(image_path)
    
    #methane_prediction = model.predict(processed_image[None])
    
    # print(methane_prediction[0])
    # print(np.array([methane_prediction[0]]).shape)

    pred = model.predict(preprocess_image(image_path)[None])[0,:,:,0]

    print("Pred min:", pred.min())
    print("Pred max:", pred.max())
    print("Pred mean:", pred.mean())
    print("Pred std:", pred.std())
    
    # get profile of base input image
    with rasterio.open(image_path) as base_img:
        base_profile = base_img.profile

        # rasterio is 1-indexed
        b = base_img.read(1)
        g = base_img.read(2)
        r = base_img.read(3)
        b11 = base_img.read(4)
        b12 = base_img.read(5)
        emit = base_img.read(6)

    # normalize rgb values to 0-255
    rgb = np.dstack((
        (((r - r.min()) / (r.max() - r.min())) * 255),
        (((g - g.min()) / (g.max() - g.min())) * 255),
        (((b - b.min()) / (b.max() - b.min())) * 255))
    ).astype('uint8')

    # create and save subplot figure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,3), layout="constrained")
        
    axes[0].imshow(
        np.log1p(pred),
        cmap="cividis"
    )
    axes[0].set_title("Predicted Methane Plume (log scale)")
    sm = plt.cm.ScalarMappable(cmap="cividis")
    fig.colorbar(sm, ax=axes[0], shrink=1)

    emit[emit < -1000] = np.nan # set no data values to nan for better visualization
    emit = np.maximum(emit, 0) # set negative values to 0

    emit_log = np.log1p(np.maximum(emit, 0))

    emit_min = np.nanpercentile(emit_log, 5)
    emit_max = np.nanpercentile(emit_log, 95)
    emit_scaled = (emit_log - emit_min) / (emit_max - emit_min) # normalize for better visualization
    emit_scaled = np.clip(emit_scaled, 0, 1) # clip to [0, 1] range for visualization

    axes[0].set_title("Predicted Methane Plume Map")
    axes[1].imshow(emit_scaled, cmap="cividis")
    axes[1].set_title('EMIT Ground Truth (log scale)')
    axes[2].imshow(rgb)
    axes[2].set_title('RGB S2 Input (Bands 2, 3, 4)')
    # axes[3].imshow(b11)
    # axes[3].set_title('SWIR-1 S2 Input (Band 11)')
    # axes[4].imshow(b12)
    # axes[4].set_title('SWIR-2 S2 Input (Band 12)')
    
    fig.savefig("EarthRemoteSensingRapidResponse/Predictions/testprediction.png")
    plt.show()
    
    # file path to save .tif to temporarily
    temp_path = "EarthRemoteSensingRapidResponse/Predictions/testprediction.tif"
        
    # save new .tif of prediction
    with rasterio.open(
        temp_path, 
        **{**base_profile, "count": 1},
        mode="w"
    ) as file:
        file.write(np.expand_dims(pred, axis=0))
        
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
    test_image_path = test_image
    errsr_model_prediction(test_image_path)

    # plot metrics
    plot_history(history, "loss")
    plot_history(history, "mean_absolute_error")
    
    return

if __name__ == "__main__":
    main()