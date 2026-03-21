import numpy as np
import matplotlib.pyplot as plt
import simplekml
import os
import sys
import rasterio
from rasterio.plot import show

# image to visualize
combined_path = f"EarthRemoteSensingRapidResponse/Data Collection/train_test_/20240206T163449_20240206T164303_T16SDB&EMIT_L2B_CH4PLM_001_20240202T165156_002642tile_0.tif"

with rasterio.open(combined_path) as image:
    data = image.read()
    profile = image.profile
    bounds = image.bounds
    transform = image.transform
    nodata = image.nodata
    
    b = image.read(1)
    g = image.read(2)
    r = image.read(3)
    
    emit = image.read(6)
    
rgb = np.dstack((
    (((r - r.min()) / (r.max() - r.min())) * 255),
    (((g - g.min()) / (g.max() - g.min())) * 255),
    (((b - b.min()) / (b.max() - b.min())) * 255))
).astype('uint8')
rgb = np.transpose(rgb, (2, 0, 1))

show(emit, cmap="inferno")
show(rgb)