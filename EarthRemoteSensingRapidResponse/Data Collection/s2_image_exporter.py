import ee
import time
import pandas as pd
import io
import os
from datetime import datetime, timedelta
import requests
import json
import shutil
import math

CLOUD_COVER_MAX = 5 # Max percentage of clouds 
DATETIME_RANGE = 90 # Start date of plume, to DATETIME_RANGE days after

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

found_s2_tifs = []

def get_s2_image_and_export(roi_to_use, start_date_str, end_date_str, output_local_dir):
    """
    Fetches and exports a Sentinel-2 image for a given point and date range.

    Args:
        longitude (float): Longitude of the target point.
        latitude (float): Latitude of the target point.
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.

    Returns true if URL was successfully downloaded,
            false otherwise
    """

    '''
    print(f"\nProcessing point: Lon={longitude}, Lat={latitude}, Date Range={start_date_str} to {end_date_str}")

    point = ee.Geometry.Point(longitude, latitude)

    # Apply getBoundingBox function to MethaneAir feature collection
    boundingBoxes = pointSourceCollection.map(getBoundingBox)

    # Convert feature geometry to region of interest
    roi = boundingBoxes.filterBounds(point).geometry() # Bottom-left corner is the given coordinate, and the coordinate that closes off the roi
    '''
    log_lon, log_lat = 'N/A', 'N/A'
    try:
        roi_info = roi_to_use.getInfo()
        if not roi_info or 'coordinates' not in roi_info or not roi_info['coordinates']:
            print(f"Provided ROI is empty or invalid.")
            return False
        
        centroid_coords = roi_to_use.centroid(1).coordinates().getInfo()
        log_lon, log_lat = centroid_coords[0], centroid_coords[1]
    except Exception as e:
        print(f"Error getting centroid for ROI: {e}")
        log_lon, log_lat = 'Error (Centroid)', 'Error (Centroid)'

    print(f"\nProcessing ROI around Lon={log_lon}, Lat={log_lat}, Date Range={start_date_str} to {end_date_str}")

    roi = roi_to_use

    # Import Sentinel-2 Imagery (harmonized)
    S2_Harmonized = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(roi) \
        .filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'LESS_THAN', CLOUD_COVER_MAX) \
        .filterDate(start_date_str, end_date_str) \
        .select(['B2', 'B3', 'B4', 'B11', 'B12'])

    # There may be overlapping images: get the first one
    S2Image = S2_Harmonized.first()

    # Check if an Earth Engine image was actually found by evaluating its info
    image_info = S2Image.getInfo()

    if image_info is None:
        print(f"No Sentinel-2 image found for the given criteria (Lon: {longitude}, Lat: {latitude}, Start: {start_date_str}, End: {end_date_str}, ROI bounds, and cloud coverage < {CLOUD_COVER_MAX}%).")
        return False    # Exit the function if no image is found

    # resample bands B2, B3, B4 (10m per pixel) to match B11, B12 (20m per pixel)
    S2ImageProjection = S2Image.select('B12').projection()
    S2ImageBands = S2Image.select('B2','B3','B4','B11','B12')
    S2ImageResample = S2ImageBands.reproject(crs=S2ImageProjection, scale=20)

    # Normalize image (Sentinel 2 imagery has values of 0-10000,
    #so divide by 10000 to get values between 0-1)
    S2ImageRescale = S2ImageResample.divide(1) # Want pure values
    clipS2Image = S2ImageRescale.clip(roi)

    # Visualization parameters (for export.visualize)
    # visParam = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
    # clipS2Image = clipS2Image.visualize(**visParam)

    download_params = {
        'format': 'GEO_TIFF',       
        'region': roi.getInfo(),         
        'dimensions': '256x256',  
        'crs': 'EPSG:4326'
    }

    try:
        download_url = clipS2Image.getDownloadURL(download_params)
        print(f"Generated download URL: {download_url}")
    except ee.EEException as e:
        print(f"Failed to generate download URL: {e}")
        return False

    os.makedirs(output_local_dir, exist_ok=True)

    output_filename = S2ImageResample.id().getInfo()
    output_filepath = os.path.join(output_local_dir, f"{output_filename}.tif")

    if output_filename in found_s2_tifs:
        return False

    print(f"Downloading image to: {output_filepath}")
    try:
        # Use requests to download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(output_filepath, 'wb') as f: # Download the image in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded successfully to {output_filepath}")
        found_s2_tifs.append(output_filename)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from URL: {e}")
        return False
    except Exception as e:
        print(f"Exception caught: {e}")
        return False

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

def generate_overlapping_grid_meters(geometry_to_cover):
    """
    Generates an array of lon, lat coordinates correspodning to 
    overlapping square grid cells that cover the given geometry, 
    performing calculations in meters using an appropriate UTM projection.

    Args:
        geometry_to_cover (ee.Geometry): The geometry to cover with grid cells.

    Returns:
        [][] - array of [[lon, lat], ... ] 
    """
    # Check if geometry is empty
    if geometry_to_cover.getInfo()['coordinates'] == []:
        print("Warning: Geometry to cover is empty. Cannot generate grid.")
        return ee.FeatureCollection([])

    # Determine UTM CRS
    centroid = geometry_to_cover.centroid(1)
    lon = centroid.coordinates().get(0).getInfo()
    lat = centroid.coordinates().get(1).getInfo()
    utm_crs = get_utm_crs_for_point(lon, lat)

    # Project the plume geometry to UTM
    projected_geometry = geometry_to_cover.transform(utm_crs, 1)
    projected_bounds = projected_geometry.bounds(1, utm_crs)

    coords_proj = projected_bounds.coordinates().getInfo()

    coords_proj = coords_proj[0]
    min_x_proj, min_y_proj = coords_proj[0]
    max_x_proj, max_y_proj = coords_proj[2]

    tile_width_meters = 2560 # 2.56km
    tile_height_meters = 2560 # 2.56km

    # Add a small buffer to bounds to ensure full coverage
    buffer_x = tile_width_meters * 0.1
    buffer_y = tile_height_meters * 0.1
    min_x_proj -= buffer_x
    min_y_proj -= buffer_y
    max_x_proj += buffer_x
    max_y_proj += buffer_y



    rois_list = []
    current_x = min_x_proj
    while current_x < max_x_proj:
        current_y = min_y_proj
        while current_y < max_y_proj:

            x1 = current_x
            y1 = current_y
            x2 = current_x + tile_width_meters
            y2 = current_y + tile_height_meters

            utm_corners = [
                [x1, y1], # Bottom-left
                [x1, y2], # Top-left
                [x2, y2], # Top-right
                [x2, y1], # Bottom-right
                [x1, y1]  # Close the polygon
            ]

            tile_rect_proj = ee.Geometry.Polygon(
                [utm_corners],
                proj=utm_crs,
                geodesic=False
            )
            
            if not tile_rect_proj.transform('EPSG:4326', 1).intersects(geometry_to_cover, 1):
                current_y += tile_height_meters
                continue
            
            # print(tile_rect_proj.bounds(1, utm_crs).coordinates().getInfo())
            # print(tile_rect_proj.transform('EPSG:4326', 1).bounds().coordinates().getInfo())

            # utm_point = ee.Geometry.Point([current_x, current_y], utm_crs)
            # wgs84_point = utm_point.transform('EPSG:4326', 1)

            # print(utm_point.getInfo())
            # print(wgs84_point.getInfo())
            
            # utm_point = ee.Geometry.Point([current_x+ tile_width_meters, current_y+ tile_height_meters], utm_crs)
            # wgs84_point = utm_point.transform('EPSG:4326', 1)

            # print(utm_point.getInfo())
            # print(wgs84_point.getInfo())
            #lon_lat_coords = wgs84_point.coordinates().getInfo() 

            tile_rect_wgs84 = tile_rect_proj.transform('EPSG:4326', 1)
            
            rois_list.append(tile_rect_wgs84)
            current_y += tile_height_meters
        current_x += tile_width_meters
    
    return rois_list

# EMIT Plume list
emit_folder_path = './EMIT_Plumes/EMITL2BCH4PLM_001-20260122_054222'

# TODO : Run the function across the entire EMIT dataset,

# Example
longitude = -97.92534000000002
latitude = 35.64508
start = '2023-06-18' 
end = '2023-6-20'
output_local_dir = "./s2"

#get_s2_image_and_export(longitude, latitude, start, end, output_local_dir)

test_output_folder = "./test/"
test_emit_no_s2_folder = "./unpaired_EMIT/"

folderIndex = 0
foundImageFolderList = []
for filename in os.listdir(emit_folder_path):
    grid_exists = False
    if filename.endswith('.json'):
        # Initialize folder for current EMIT plume
        outputFolder = os.path.join(test_output_folder, str(folderIndex))
        os.makedirs(outputFolder, exist_ok=True)

        # Get JSON metadata
        json_path = os.path.join(emit_folder_path, filename)
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Just testing the first coordinate. Can experiment and try different points within the geometry
        coordinates = data["features"][0]['geometry']['coordinates'][0]
        firstCoord = coordinates[0]
        firstCoordLong = firstCoord[0]
        firstCoordLat = firstCoord[1]


        # Dates
        observedTime = data["features"][0]['properties']['UTC Time Observed']
        dt_obj = datetime.strptime(observedTime, "%Y-%m-%dT%H:%M:%SZ").date()
        startDate = dt_obj.isoformat()
        endDate = (dt_obj + timedelta(days=DATETIME_RANGE)).isoformat()

        # Get meter-based tiles
        plume_json_geometry = data["features"][0]['geometry']
        plume_ee_geometry = ee.Geometry(plume_json_geometry)

        roi_array = generate_overlapping_grid_meters(plume_ee_geometry)

        if not roi_array:
            print("No grids genereated for this EMIT plume")
            continue

        print("Current Plume: ", firstCoordLong, " : ", firstCoordLat)

        for i, roi in enumerate(roi_array):



            outputFolder = os.path.join(test_output_folder, str(folderIndex), f"grid_{i+1}")
            os.makedirs(outputFolder, exist_ok=True)
            
            # Get the related .tif file
            parts = filename.split("META")
            tif_file = parts[0] + parts[1].replace('json', 'tif')
            
            tif_path_orig = os.path.join(emit_folder_path, tif_file)
            tif_path_new = os.path.join(outputFolder, tif_file)

            # Copy .tif file to output folder
            shutil.copy(tif_path_orig, tif_path_new)

            found = get_s2_image_and_export(roi, 
                                    startDate, 
                                    endDate,
                                    outputFolder)
        
            if found:
                foundImageFolderList.append(outputFolder)
                grid_exists = True
            else:
                os.remove(tif_path_new)
                os.rmdir(outputFolder)
    
        if not grid_exists:
            outputFolder = os.path.join(test_output_folder, str(folderIndex))
            os.rmdir(outputFolder)

            # Folder for unpaired emit plumes
            os.makedirs(test_emit_no_s2_folder, exist_ok=True)

            tif_path_new = os.path.join(test_emit_no_s2_folder, tif_file)
            # Copy .tif file to output folder
            shutil.copy(tif_path_orig, tif_path_new)
        
        folderIndex += 1

print("Folders with images: ", foundImageFolderList)
print("Count: ", len(foundImageFolderList))