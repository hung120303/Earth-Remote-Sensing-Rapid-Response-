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

# np.set_printoptions(threshold=sys.maxsize)

# define file paths
imageset = []
names = []
selected_set = 1

dataset = f"./UnprocessedImages/{selected_set}"
for file in os.listdir(dataset):
    path = os.path.join(dataset, file)
        
    # open image
    if os.path.isfile(path):
        names.append(file)
        imageset.append(path)

s2_path = imageset[0]
emit_path = imageset[1]

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

# print sizes of each array
print(np.array(s2_data).shape)
print(np.array(emit_data).shape)
print(np.array(array).shape)

# combine s2 and emit data into one array
combined_data = np.concatenate((s2_data, array), axis=0)
print(combined_data.shape)

# visualize data
show(emit_data, transform=emit_transform, cmap="inferno")
show(array, transform=emit_transform, cmap="inferno")
show(s2_data[1], transform=s2_transform)

# save combined array as a new file
with rasterio.open(
    f"./Dataset/validation/{names[0]}", 
    **{**s2_profile, "count": 6},
    mode="w"
) as file:
    for band, data in enumerate(combined_data, start=1):
        file.write(data, band)