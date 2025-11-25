import numpy as np
import matplotlib.pyplot as plt
import simplekml
import os
import sys

# Test file for working with and visualizing the geotiff files to make sure they are suitable as input. 

# Testing libraries for working with tif
import cv2
import rasterio
from rasterio.plot import show

# np.set_printoptions(threshold=sys.maxsize)

# define file paths
s2_path = "EarthRemoteSensingRapidResponse/TempImageSet/20230203T171519_20230203T172113_T14RMU.tif"
emit_path = "EarthRemoteSensingRapidResponse/TempImageSet/EMIT_L2B_CH4PLM_001_20230206T162514_000635.tiff"

test_path = "EarthRemoteSensingRapidResponse/TempImageSet/1-20230203T171519_20230203T172113_T14RMU.tif"

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
    
# s2_profile.update(nodata=s2_nodata)

print(np.array(s2_data).shape)
print(np.array(emit_data).shape)
print(np.array(array).shape)

show(emit_data)
show(array)

# Had issues displaying the 5 band s2 image, so I'm temporarily using the composite image for displaying the plume location
with rasterio.open(test_path) as test_image:
    test_data = test_image.read()
    
show(test_data)
    
# overlay plume on s2 image
fig, ax = plt.subplots(figsize=(5,5))
show(test_data, ax=ax)
show(array, ax=ax, alpha=0.5)
plt.show()