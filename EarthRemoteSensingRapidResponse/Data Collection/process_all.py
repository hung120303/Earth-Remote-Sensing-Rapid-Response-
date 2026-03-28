####################################################################################################
# ImagePreprocessing.py                                                                            #
#   - This program file reprojects the given EMIT data to match the geospatial coordinates         #
#     of the S2_Harmonized input data, and combines the files together into one .tif with 6 bands. #
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import simplekml
import os
import sys
import rasterio
from rasterio.plot import show
import time

# np.set_printoptions(threshold=sys.maxsize)

# Set to true to scan through each of the datapoints
INSPECT_IMAGES = False
RETRIEVE_GOOD = True # condition to only download good images to the folder

folder_path = "/3_19_2026_test"

output_folder_path = "/train_test_"

# define file paths
names = []
s2_filepaths = []
emit_filepaths = []
final_tif_output_filepaths = []
badFiles = []

dataset = f"EarthRemoteSensingRapidResponse/Data Collection/{folder_path}"
for path, subfolders, file in os.walk(dataset):
    if file:
        if len(file) != 2:
            continue
        print(path)
        s2_path = os.path.join(path, file[0])
        emit_path = os.path.join(path, file[1])
        grid_name = path.split('\\')
        new_name = (file[0]).replace(".tif", "") + "_" +(file[1]).replace(".tif", "") + grid_name[-1] + ".tif"

        s2_filepaths.append(s2_path)
        emit_filepaths.append(emit_path)
        names.append(new_name)

n = len(names)
print(f"Processing {n} datapoints: ")

bad_emit_count = 0
bad_s2_count = 0
bad_pair_count = 0 

bad_emit_max_vals = []

for i in range(len(names)):
    print((i+1), names[i])
    s2_path = s2_filepaths[i]
    emit_path = emit_filepaths[i]

    # open images and copy relevant data
    with rasterio.open(s2_path) as s2_image:
        s2_data = s2_image.read()
        s2_profile = s2_image.profile
        s2_bounds = s2_image.bounds
        s2_transform = s2_image.transform
        s2_nodata = s2_image.nodata

    with rasterio.open(emit_path) as emit_image:
        emit_data = emit_image.read()
        emit_profile = emit_image.profile
        emit_bounds = emit_image.bounds
        emit_transform = emit_image.transform

    # define a window based on the s2 image: this will be used to resample the EMIT plume
    window = rasterio.windows.from_bounds(*s2_bounds, transform=emit_transform)

    #print("S2: ",*s2_bounds)
    #print("EMIT: ",emit_bounds)

    # resampling of EMIT plume
    with rasterio.open(emit_path) as emit_image:
        emit_nodata = emit_image.nodata
        array = emit_image.read(
            window=window, 
            boundless=True,
            out_shape=(
                emit_image.count,
                int(256),
                int(256),
            ),
            resampling=rasterio.enums.Resampling.gauss
        )

    if INSPECT_IMAGES:
        # print sizes of each array
        print(np.array(s2_data).shape)
        print(np.array(emit_data).shape)
        print(np.array(array).shape)

    # combine s2 and emit data into one array
    combined_data = np.concatenate((s2_data, array), axis=0)
    
    if INSPECT_IMAGES:
        print(combined_data.shape)

        # visualize data
        show(emit_data)
        show(array)
        for i in range(s2_data.shape[0]):
            show(s2_data[i])

    # Bad data checking
    S2_MIN_PERCENTAGE = 0.0
    S2_MAX_NODATA_RATIO = 1 - S2_MIN_PERCENTAGE
    MIN_EMIT_PPM_VAL = 300
    emit_max_plume_val = np.max(np.array(array)) # Check if the max value of resampled plume is 0 (rest of the image should be 0)
    
    s2_mask = np.any(s2_data == s2_nodata, axis=0)
    nodata_ratio = np.mean(s2_mask)

    bad_emit = False
    bad_s2 = False

    if(emit_max_plume_val < 0): # Plumes with -9999 as the max have purple across the baord
        print("Problem with EMIT re-sampling: ", emit_path)
        #bad_emit_max_vals.append(emit_max_plume_val)
        bad_emit = True
        bad_emit_count += 1
    elif emit_max_plume_val < MIN_EMIT_PPM_VAL: # These plumes did have some ppm visible 
        print("") # 
        #show(array)
    
    if(nodata_ratio > S2_MAX_NODATA_RATIO):
        print("Problem with s2 image: ", s2_path)
        print("No data ratio: ", nodata_ratio)
        bad_s2 = True
        bad_s2_count += 1
        #show(s2_data[3])

    if (bad_emit or bad_s2) and RETRIEVE_GOOD:
        bad_pair_count += 1
        continue # skip if we want only the good ones


    final_tif_output_filepaths.append(f"EarthRemoteSensingRapidResponse/Data Collection/{output_folder_path}{names[i]}")
    # save combined array as a new file
    with rasterio.open(
        f"EarthRemoteSensingRapidResponse/Data Collection/{output_folder_path}/{names[i]}", 
        **{**s2_profile, "count": 6},
        mode="w"
    ) as file:
        for band, data in enumerate(combined_data, start=1):
            file.write(data, band)

print("Bad emits: ",bad_emit_count)
print("Bad s2s: ", bad_s2_count)
print("Bad pairs: ",bad_pair_count)



