import numpy as np
import matplotlib.pyplot as plt
import simplekml
import os
import sys

# Test file for working with and visualizing the geotiff files to make sure they are suitable as input. 
# Also used for processing and combining S2 and EMIT data into one raster

# Testing libraries for working with tif
import cv2
import rasterio
from rasterio.plot import show

# np.set_printoptions(threshold=sys.maxsize)

# define file paths
imageset = []
names = []
selected_set = 1

dataset = f"EarthRemoteSensingRapidResponse/UnprocessedImages/{selected_set}"
for file in os.listdir(dataset):
    path = os.path.join(dataset, file)
        
    # open image
    if os.path.isfile(path):
        names.append(file)
        imageset.append(path)
    
# s2_path = "EarthRemoteSensingRapidResponse/UnprocessedImages/20230203T171519_20230203T172113_T14RMU.tif"
# emit_path = "EarthRemoteSensingRapidResponse/UnprocessedImages/EMIT_L2B_CH4PLM_001_20230206T162514_000635.tiff"
# ugh_path = "EarthRemoteSensingRapidResponse/UnprocessedImages/2-20230203T171519_20230203T172113_T14RMU.tif"
# test_path = "EarthRemoteSensingRapidResponse/UnprocessedImages/test.tif"

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

show(emit_data)
show(array)
show(s2_data[1])

# with rasterio.open(ugh_path) as ugh:
#     ugh_data = ugh.read()
    
# print(ugh_data)
# print(s2_nodata)
# print(emit_data[emit_data != -9999])
# print(array[array != -9999])
# print(combined_data[5][combined_data[5] != -9999])

# with rasterio.open(test_path) as test_image:
#     test_data = test_image.read()
    
# print(s2_data)
    
# print(np.array(test_data).shape)

# save combined array as a new file (uncomment)
with rasterio.open(
    f"EarthRemoteSensingRapidResponse/Dataset/train_test/{names[0]}", 
    **{**s2_profile, "count": 6},
    mode="w"
) as file:
    for band, data in enumerate(combined_data, start=1):
        file.write(data, band)

# Had issues displaying the 5 band s2 image, so I'm temporarily using the composite image for displaying the plume location
# with rasterio.open(test_path) as test_image:
#     test_data = test_image.read()
    
# show(test_data)
    
# overlay plume on s2 image
# fig, ax = plt.subplots(figsize=(5,5))
# show(test_data, ax=ax)
# show(array, ax=ax, alpha=0.5)
# plt.show()