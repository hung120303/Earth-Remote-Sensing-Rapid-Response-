import pandas as pd
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np


###
# This program is my first iteration of determining if the coordinates of Iowa CAFO's
# are inside the boundaries of a created polygon.
#

# Just Iowa here, but the recursive will be used for multiple csv's
CAFO_csv = glob.glob('**/*iowa_cafos_2024_arcgis_api.csv', recursive=True)

dfs = []

# Recursively appends for all csv files
# For now, not necessary because Iowa only has one csv.
for file in CAFO_csv:
    df = pd.read_csv(file)
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    df = df.fillna('')  # Replace NaN with empty string
    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)
master_df.to_csv('Master_arcgis.csv', index=False)

# Convert to numeric & cleaning df
master_df['Longitude'] = pd.to_numeric(master_df['Longitude'], errors='coerce')
master_df['Latitude'] = pd.to_numeric(master_df['Latitude'], errors='coerce')

master_df = master_df.dropna(subset=['Longitude', 'Latitude'])

# Define some random polygon,
polygon = np.array([
    [-95.0, 41],  # Point 1 (Lon, Lat)
    [-91.0, 42],
    [-91.0, 43.0],
    [-95.0, 43.0] ,
    [-99.0, 44.0] # Can add more points if you wish
], dtype=np.float32)

# We'll be dividing the coordinates to those inside the polygon
# And those outside the polygon
inside_coord = []
outside_coord = []

# using the opencv2 library to checks if coordinates are either inside or outside the polygon
for idx, row in master_df.iterrows():
    point = (float(row['Longitude']), float(row['Latitude']))
    distance = cv2.pointPolygonTest(polygon, point, False)

    if distance >= 0:
        inside_coord.append(row)
    else:
        outside_coord.append(row)

# Categorizing coordinates to their respective df
inside_df = pd.DataFrame(inside_coord)
outside_df = pd.DataFrame(outside_coord)


# Plotting
plt.figure(figsize=(12, 8))

# Plot outside coordinates
if len(outside_df) > 0:
    plt.scatter(outside_df['Longitude'], outside_df['Latitude'],
                c='blue', alpha=0.5, label='Outside polygon', s=30)

# Plot inside coordinates
if len(inside_df) > 0:
    plt.scatter(inside_df['Longitude'], inside_df['Latitude'],
                c='red', alpha=0.7, label='Inside polygon', s=50)


# Draw the polygon
polygon_closed = np.vstack([polygon, polygon[0]])  # Close the polygon
plt.plot(polygon_closed[:, 0], polygon_closed[:, 1],
         'g-', linewidth=2, label='Polygon boundary')
plt.fill(polygon[:, 0], polygon[:, 1], 'green', alpha=0.1)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary
print("\n=== Summary ===")
print(f"Total points: {len(master_df)}")
print(f"Points inside polygon: {len(inside_df)}")
print(f"Points outside polygon: {len(outside_df)}")