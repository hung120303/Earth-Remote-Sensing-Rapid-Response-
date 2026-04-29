import rasterio
import os
import csv
import shutil

'''
attach_cafo.py

Outputs folder of images that contain a CAFO lon, lat point, with folder output from process_all.py 

'''

s2_emit_combined_image_input_folder = "/train_test_s2_0"
output_folder = "/images_with_cafo"
output_folder_non_cafo = "/images_non_cafo"
cafos_folder = "/cafo_csv"

def containsCAFO(bounds, cafo_points):
    for lon, lat in cafo_points:
        if (bounds.left <= lon <= bounds.right and
            bounds.bottom <= lat <= bounds.top):
            return True
    return False

def load_cafo_points(csv_path):
    points = []

    lat_names = {'lat', 'latitude', 'latdec', 'lat_facili', 'y'}
    lon_names = {'lon', 'long', 'longitude', 'londec', 'lon_facili', 'x'}


    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Normalize column names (lowercase + strip spaces)
        field_map = {col.lower().strip(): col for col in reader.fieldnames}

        lat_col = None
        lon_col = None

        # Detect latitude column
        for lat_name in lat_names:
            if lat_name in field_map:
                lat_col = field_map[lat_name]
                break

        # Detect longitude column
        for lon_name in lon_names:
            if lon_name in field_map:
                lon_col = field_map[lon_name]
                break

        if lat_col is None or lon_col is None:
            print(f"Could not detect lat/lon columns in {csv_path}")
            return points

        for row in reader:
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])

                # Validate coordinate ranges
                if not (-90 <= lat <= 90):
                    continue
                if not (-180 <= lon <= 180):
                    continue

                points.append((lon, lat))  # rasterio uses (x=lon, y=lat)

            except (ValueError, TypeError):
                # Skip rows with invalid/missing values
                continue
    return points


# Get all the cafo points in a list
cafo_list = []

for path, subfolders, files in os.walk(f"EarthRemoteSensingRapidResponse/Data Collection/{cafos_folder}"):
    for file in files:
        if file.endswith(".csv"):
            cafo_path = os.path.join(path, file)
            points = load_cafo_points(cafo_path)
            cafo_list.extend(points)

cafo_path = os.path.join((f"EarthRemoteSensingRapidResponse/Data Collection/{output_folder}")
non_cafo_path = os.path.join((f"EarthRemoteSensingRapidResponse/Data Collection/{output_folder_non_cafo}")
# Check if the cafo points lay within the image
for path, subfolders, files in os.walk(f"EarthRemoteSensingRapidResponse/Data Collection/{s2_emit_combined_image_input_folder}"):
    for file in files:
        if file.endswith(".tif"):  # or whatever format
            image_path = os.path.join(path, file)

            with rasterio.open(image_path) as src:
                bounds = src.bounds

                if containsCAFO(bounds, cafo_list):
                    print(f"{file} contains a CAFO")
                    output_path = os.join(cafo_path, file)
                else:
                    print(f"{file} does NOT contain a CAFO")
                    output_path = os.join(non_cafo_path, file)
                
                shutil.copy(image_path, output_path)

