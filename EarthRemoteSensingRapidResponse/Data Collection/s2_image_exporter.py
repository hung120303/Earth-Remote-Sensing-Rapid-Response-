import ee
import time
import pandas as pd
import io
import os
from datetime import datetime, timedelta
import re
import requests

ee.Authenticate() 

ee.Initialize(project='ersrr-475700') # Ensure this project ID is correct and you have access

print("Earth Engine initialized successfully.")

# Import MethaneAir L4 Point Sources
pointSourceCollection = ee.FeatureCollection("EDF/MethaneSAT/MethaneAIR/L4point")

# Define a function for obtaining a bounding box around each feature
def getBoundingBox(feature):
    buffer = feature.buffer(2530) # radius of roughly 5120 meters (5.12km)
    box = buffer.bounds()
    return box

def get_s2_image_and_export(longitude, latitude, start_date_str, end_date_str, output_local_dir):
    """
    Fetches and exports a Sentinel-2 image for a given point and date range.

    Args:
        longitude (float): Longitude of the target point.
        latitude (float): Latitude of the target point.
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.
    """
    print(f"\nProcessing point: Lon={longitude}, Lat={latitude}, Date Range={start_date_str} to {end_date_str}")

    point = ee.Geometry.Point(longitude, latitude)

    # Apply getBoundingBox function to MethaneAir feature collection
    boundingBoxes = pointSourceCollection.map(getBoundingBox)

    # Convert feature geometry to region of interest
    roi = boundingBoxes.filterBounds(point).geometry()

    # Import Sentinel-2 Imagery (harmonized)
    S2_Harmonized = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(roi) \
        .filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'LESS_THAN', 5) \
        .filterDate(start_date_str, end_date_str) \
        .select(['B2', 'B3', 'B4', 'B11', 'B12'])

    # There may be overlapping images: get the first one
    S2Image = S2_Harmonized.first()

    # Check if an Earth Engine image was actually found by evaluating its info
    image_info = S2Image.getInfo()

    if image_info is None:
        print(f"No Sentinel-2 image found for the given criteria (Lon: {longitude}, Lat: {latitude}, Start: {start_date_str}, End: {end_date_str}, ROI bounds, and cloud coverage < 5%).")
        return # Exit the function if no image is found

    # resample bands B2, B3, B4 (10m per pixel) to match B11, B12 (20m per pixel)
    S2ImageProjection = S2Image.select('B12').projection()
    S2ImageBands = S2Image.select('B2','B3','B4','B11','B12')
    S2ImageResample = S2ImageBands.reproject(crs=S2ImageProjection, scale=20)

    # Normalize image (Sentinel 2 imagery has values of 0-10000,
    #so divide by 10000 to get values between 0-1)
    S2ImageRescale = S2ImageResample.divide(10000)
    clipS2Image = S2ImageRescale.clip(roi)

    # Visualization parameters (for export.visualize)
    visParam = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
    clipS2Image = clipS2Image.visualize(**visParam)

    download_params = {
        'format': 'GEO_TIFF',       
        'region': roi.getInfo(),         
        'dimensions': '256x256',    
    }

    try:
        download_url = clipS2Image.getDownloadURL(download_params)
        print(f"Generated download URL: {download_url}")
    except ee.EEException as e:
        print(f"Failed to generate download URL: {e}")
        return

    os.makedirs(output_local_dir, exist_ok=True)

    output_filename = S2ImageResample.id().getInfo()
    output_filepath = os.path.join(output_local_dir, f"{output_filename}.tif")

    print(f"Downloading image to: {output_filepath}")
    try:
        # Use requests to download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(output_filepath, 'wb') as f: # Download the image in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded successfully to {output_filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from URL: {e}")
    except Exception as e:
        print(f"Exception caught: {e}")

# EMIT Plume list
emit_folder_path = '.\EMIT_Plumes\EMITL2BCH4PLM_001-20260122_054222'

# TODO : Run the function across the entire EMIT dataset,

# Example
longitude = -97.92534000000002
latitude = 35.64508
start = '2023-06-18' 
end = '2023-6-20'
output_local_dir = "./s2"

get_s2_image_and_export(longitude, latitude, start, end, output_local_dir)
