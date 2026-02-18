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
from rasterio.mask import mask
from rasterio.warp import transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from rasterio.transform import from_origin
import math
import time
import ee

ee.Authenticate() 
ee.Initialize(project='ersrr-475700') # Ensure this project ID is correct and you have access

def get_utm_crs_for_point(longitude, latitude):
    """
    Determines the EPSG code for the appropriate UTM projection for a given point.
    WGS is recorded in degrees, while UTM (Universal Transverse Mercator) uses meters. 
    """
    utm_zone = int((longitude + 180) / 6) + 1
    if latitude >= 0:
        # Northern hemisphere (e.g., EPSG:326XX)
        return f'EPSG:326{utm_zone}'
    else:
        # Southern hemisphere (e.g., EPSG:327XX)
        return f'EPSG:327{utm_zone}'

# np.set_printoptions(threshold=sys.maxsize)

# Set to true to scan through each of the datapoints
INSPECT_IMAGES = False
RETRIEVE_GOOD = True # condition to only download good images to the folder

folder_path = "/2_16_2026_s2_emit_pairs_bounded_by_full_plume"

output_folder_path = "/train_test_2_17_"

# define file paths
names = []
s2_filepaths = []
emit_filepaths = []
final_tif_output_filepaths = []
badFiles = []

dataset = f"EarthRemoteSensingRapidResponse/Data Collection/{folder_path}"
for path, subfolders, file in os.walk(dataset):
    if file:
        if len(file) < 2:
            continue
        s2_path = os.path.join(path, file[0])
        emit_path = os.path.join(path, file[1])
        new_name = (file[0]).replace(".tif", "") + "&" +(file[1]).replace(".tif", "")
        
        s2_filepaths.append(s2_path)
        emit_filepaths.append(emit_path)
        names.append(new_name)
        print(new_name)

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

    # Readjust EMIT to S2
    emit_file_parts = emit_path.split('.tif')
    emit_output_path = emit_file_parts[0] + '_adjusted_' + '.tif'
    
    with rasterio.open(s2_path) as s2_image:
        s2_crs = s2_image.crs
        s2_transform = s2_image.transform
        s2_width = s2_image.width
        s2_height = s2_image.height

        with rasterio.open(emit_path) as emit_image:
            kwargs = emit_image.meta.copy()
            kwargs.update({
                'crs': s2_crs,
                'transform': s2_transform,
                'width': s2_width,
                'height': s2_height
            })

            # Download to folder
            with rasterio.open(emit_output_path, 'w', **kwargs) as dst:
                
                for j in range(1, emit_image.count + 1):
                    reproject(
                        source=rasterio.band(emit_image, j),
                        destination=rasterio.band(dst, j),
                        src_transform=emit_image.transform,
                        src_crs=emit_image.crs,
                        dst_transform=s2_transform,
                        dst_crs=s2_crs,
                        resampling=Resampling.bilinear
                    )

    with rasterio.open(emit_output_path) as emit_image:
        emit_data = emit_image.read()
        emit_bounds = emit_image.bounds
        emit_crs = emit_image.crs
    with rasterio.open(s2_path) as s2_image:
        s2_data = s2_image.read()
        s2_bounds = s2_image.bounds
        s2_profile = s2_image.profile
        s2_profile["bounds"] = s2_bounds
        s2_crs = s2_image.crs


    # Make sure bounds for new emit and s2 are equal
    if(emit_bounds != s2_bounds):
        print("Something went wrong with readjusting: bounds aren't the same")
        continue
    if(emit_crs != s2_crs):
        print("Something went wrong with readjusting: crs aren't the same")
        continue

    # Check the shapes
    if(np.array(emit_data).shape[1] != np.array(s2_data).shape[1] or np.array(emit_data).shape[2] != np.array(s2_data).shape[2]):
        print("Something went wrong with readjusting: shapes aren't the same")
        continue

    # combine s2 and emit data into one array
    combined_data = np.concatenate((s2_data, emit_data), axis=0)

    # Determine UTM CRS
    min_x_emit, min_y_emit, max_x_emit, max_y_emit = emit_bounds
    emit_corners = [
                [min_x_emit, min_y_emit], # Bottom-left
                [min_x_emit, max_y_emit], # Top-left
                [max_x_emit, max_y_emit], # Top-right
                [max_x_emit, min_y_emit], # Bottom-right
                [min_x_emit, min_y_emit]  # Close the polygon
            ]
    emit_polygon_bounds = ee.Geometry.Polygon(emit_corners)
    emit_centroid = emit_polygon_bounds.centroid(1)
    lon = emit_centroid.coordinates().get(0).getInfo()
    lat = emit_centroid.coordinates().get(1).getInfo()
    utm_crs = get_utm_crs_for_point(lon, lat)
    
    # Use utm to transform image
    transform, width, height = calculate_default_transform(
        s2_profile["crs"],   
        utm_crs,             
        s2_profile["width"],
        s2_profile["height"],
        *s2_profile["bounds"]
    )

    left, bottom, right, top = rasterio.warp.transform_bounds(
        s2_profile["crs"], utm_crs, *s2_profile["bounds"]
    )

    # Get the right pixel resolution
    desired_resolution = 20 # 20m pixels
    width = math.ceil((right - left) / desired_resolution)
    height = math.ceil((top - bottom) / desired_resolution)
    transform = from_origin(
        left,
        top,
        desired_resolution,
        desired_resolution
    )

    utm_profile = s2_profile.copy()
    utm_profile.update({
        "crs": utm_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": combined_data.shape[0]
    })


    combined_image_folder_path = "combined_s2_emit_full_not_gridded"
    combined_image_path = f"EarthRemoteSensingRapidResponse/Data Collection/{combined_image_folder_path}/{names[i]}.tif"
    print(combined_image_path)
    with rasterio.open(combined_image_path, "w", **utm_profile) as dst:
        for band_index in range(combined_data.shape[0]):
            reproject(
                source=combined_data[band_index],
                destination=rasterio.band(dst, band_index + 1),
                src_transform=s2_profile["transform"],
                src_crs=s2_profile["crs"],
                dst_transform=transform,
                dst_crs=utm_crs,
                resampling=Resampling.bilinear
            )
    

    ##NOTE: Some of the s2 images that are bounded by the EMIT are smaller than 512x512
    ##      This will only get the data for EMIT plumes bigger than 512x512
    ##TODO: In process_all.py, we need to check the boundingboxs that are smaller, then 
    ##      download an s2 that is of a 512x512 size that contains the EMIT

    with rasterio.open(combined_image_path) as combined_image:
        tile_size = 5120 # 5120m
        
        x_resolution = combined_image.transform[0]
        y_resolution = -combined_image.transform[4]

        tile_width = int(tile_size / x_resolution)
        tile_height = int(tile_size / y_resolution)

        image_width = combined_image.width
        image_height = combined_image.height

        tile_id = 0
        for row in range(0, image_width, tile_width):
            for col in range(0, image_height, tile_height):
                # oob
                width = min(tile_width, image_width - col)
                height = min(tile_height, image_height - row)

                if width < tile_width or height < tile_height:
                    continue


                window = Window(col, row, width, height)

                transform = combined_image.window_transform(window)

                show(s2_data[3])
                data = combined_image.read(window=window)
                show(data[3])
                # Bad data checking
                VALID_PIXELS_S2 = 256*256
                MIN_EMIT_PPM_VAL = 300
                emit_max_plume_val = np.max(np.array(data[5])) # Check if the max value of resampled plume is 0 (rest of the image should be 0)
                nonzero_s2_value_count = np.count_nonzero(np.array(data[3])) # Count number of nonzero (valid) s2 pixels
                bad_emit = False
                bad_s2 = False

                if(emit_max_plume_val < 0): # Plumes with -9999 as the max have purple across the baord
                    print("Problem with EMIT re-sampling: ", emit_path)
                    #bad_emit_max_vals.append(emit_max_plume_val)
                    bad_emit = True
                    bad_emit_count += 1
                elif emit_max_plume_val < MIN_EMIT_PPM_VAL: # These plumes did have some ppm visible 
                    print("") # 
                    #show(data[5])

                
                if(nonzero_s2_value_count < VALID_PIXELS_S2):
                    print("Problem with s2 image: ", s2_path)
                    bad_s2 = True
                    bad_s2_count += 1
                    #show(data[3])

                if (bad_emit or bad_s2) and RETRIEVE_GOOD:
                    continue # skip if we want only the good ones
                    bad_pair_count += 1

                meta = combined_image.meta.copy()
                meta.update({
                    "height": height,
                    "width": width,
                    "transform": transform
                })

                output_path = f"EarthRemoteSensingRapidResponse/Data Collection/{output_folder_path}/{names[i]}tile_{tile_id}.tif"
                
                print("Saving to: ", output_path)
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(data)

                tile_id +=1

# Need to update this
print("Bad emits: ",bad_emit_count)
print("Bad s2s: ", bad_s2_count)
print("Bad pairs: ",bad_pair_count)



