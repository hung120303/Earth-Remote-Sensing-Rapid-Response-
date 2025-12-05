import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

###
# This file gets the CAFO csv's (Consisting of metedata from CAFO's in the midwest), creates one random polygon & plots them
# It will determine which coordinates are inside a polygon area and divides which coordinates
# inside the polygon & outside the polygon to their respective df
###

# Concats all the csv files into a master csv
# For now, we just have one csv. But it will work for more
files = glob.glob('**/*.csv', recursive=True)
dfs = [pd.read_csv(f).dropna(how='all').drop_duplicates() for f in files]

master_df = pd.concat(dfs, ignore_index=True).fillna('')

# Not necessary, but converts to coordinates to numeric as insurance
cols = ['Longitude', 'Latitude']
for col in cols:
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

master_df = master_df.dropna(subset=cols)

# We're using CAFO's Iowa in this example
# Max polygon size will be within the borders of Iowa
MIN_LON, MAX_LON = -95, -91
MIN_LAT, MAX_LAT = 41, 43

## Just randomizing the polygon sizes every time the program runs.
width = np.random.uniform(0.5, 2.0)
height = np.random.uniform(0.5, 1.5)

x_start = np.random.uniform(MIN_LON, MAX_LON - width)
y_start = np.random.uniform(MIN_LAT, MAX_LAT - height)

polygon_coords = [
    [x_start, y_start],           # Bottom-Left
    [x_start + width, y_start],   # Bottom-Right
    [x_start + width, y_start + height], # Top-Right
    [x_start, y_start + height]   # Top-Left
]

# Vectorized Spatial Filtering
# Creating two dataframes for coordinates inside & outside the polygon
poly_path = Path(polygon_coords)
points = master_df[['Longitude', 'Latitude']].values
mask = poly_path.contains_points(points)

inside_df = master_df[mask]
outside_df = master_df[~mask]

inside_df.to_csv('points_inside.csv', index=False)
outside_df.to_csv('points_outside.csv', index=False)

# Plotting everything
plt.figure(figsize=(12, 8))

if not outside_df.empty:
    plt.scatter(outside_df.Longitude, outside_df.Latitude, c='blue', alpha=0.5, s=30, label='Outside')

if not inside_df.empty:
    plt.scatter(inside_df.Longitude, inside_df.Latitude, c='red', alpha=0.7, s=50, label='Inside')

# Draw Polygon
patch = plt.Polygon(polygon_coords, closed=True, color='green', alpha=0.1, ec='green', lw=2, label='Boundary')
plt.gca().add_patch(patch)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#print(f"Total: {len(master_df)} | Inside: {len(inside_df)} | Outside: {len(outside_df)}")