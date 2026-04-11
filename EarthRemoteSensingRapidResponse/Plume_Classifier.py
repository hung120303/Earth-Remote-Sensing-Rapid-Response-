##########################################################################################################################
# Plume_Classifier.py                                                                                                    #
#  - This program file converts the predicted plume to a binary mask to find the point source of the plume, retrieve the # 
#    coordinates of the point source, and then compares to the CAFO database to evaluate the accuracy of the model.      #
#    The model will evaluate using true positives, false positives, and false negatives.                                 #   
##########################################################################################################################

import numpy as np
import pandas as pd
from scipy import ndimage
import rasterio
from geopy.distance import geodesic
import matplotlib.pyplot as plt

THRESHOLD_VALUE = 0.15 # Threshold value for binary mask. May adust based on model confience
DISTANCE_THRESHOLD = 1.0 # Distance threshold in kilometers for matching predicted plume source to CAFO locations
FILTER_SIZE = 50 # Minimum size of plume region in pixels to be considered valid

test_image = "EarthRemoteSensingRapidResponse/Dataset/validation/20240613T090559_20240613T091055_T34RET_EMIT_L2B_CH4PLM_001_20240612T135823_003244grid_1.tif"

def binary_mask(pred):
    mask = pred > THRESHOLD_VALUE
    return mask

def detect_plume_region(mask):
    labels, num_features = ndimage.label(mask)
    return labels, num_features

def region_filtering(mask, labels, num_features):
    sizes = ndimage.sum(mask, labels, range(1, num_features + 1))

    valid_regions = []
    for i,s in enumerate(sizes):
        if s >= FILTER_SIZE:
            valid_regions.append(i + 1)

    return valid_regions

def filter_by_distance(points, center, maxkm=5):
    filtered = []

    for p in points:
        if(geodesic(center,p).km <= maxkm):
            filtered.append(p)

    return filtered

def compute_image_center(image):
    with rasterio.open(image) as src:
        lon, lat = src.xy(src.height//2, src.width//2)
    return (lon, lat)



def compute_peaks(pred, labels, region):
    peaks = []
    for id in region:
        region_mask = (labels == id)
        val = pred * region_mask

        index = np.argmax(val)
        row, col = np.unravel_index(index,pred.shape)
        peaks.append((row,col))
    return peaks

def px_to_coords(row, col, transform):
    x, y = rasterio.transform.xy(transform, row, col)
    return (y, x)

def plume_to_coords(plume, transform):
    coords = []

    for r,c in plume:
        coord = px_to_coords(int(r), int(c), transform)
        coords.append(coord)

    return coords

def load_CAFO(csv):
    df = pd.read_csv(csv)
    cafo_coords = list(zip(df['Latitude'], df['Longitude']))
    return cafo_coords

def match_plume_to_CAFO(plume_coords, cafo_coords):

    tp = 0
    fp = 0
    matched_gt = set()

    for plume in plume_coords:
        found_match = False

        for i, gt in enumerate(cafo_coords):
            distance = geodesic(plume, gt).km
            if distance <= DISTANCE_THRESHOLD:
                found_match = True
                tp += 1
                matched_gt.add(i)
                break
        if not found_match:
            fp += 1
    fn = len(cafo_coords) - len(matched_gt)
    return tp, fp, fn

def compute_metrics(tp, fp, fn):

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def evaluate_model(pred, cafo_csv, raster_path):

    mask = binary_mask(pred)
    labels, num_features = detect_plume_region(mask)

    filtered_regions = region_filtering(mask, labels, num_features)
    peaks = compute_peaks(pred, labels, filtered_regions)

    with rasterio.open(raster_path) as src:
        transform = src.transform
    
    predicted_sources = plume_to_coords(peaks, transform)

    image_center = compute_image_center(raster_path)
    cafo_sources = filter_by_distance(load_CAFO(cafo_csv), image_center)

    plot_sources(pred, peaks, cafo_sources, transform)

    tp, fp, fn = match_plume_to_CAFO(predicted_sources, cafo_sources)

    precision, recall, f1_score = compute_metrics(tp, fp, fn)

    return{
        "True Positives" : tp,
        "False Positives" : fp,
        "False Negatives" : fn,
        "Precision" : precision,
        "Recall" : recall,
        "F1_Score" : f1_score
    }

def plot_sources(pred, pred_pixel, cafo_coords, transform, title="Plume Source Detection"):
    plt.figure(figsize=(8,8))

    plt.imshow(pred, cmap="cividis")
    plt.colorbar(label="prediction")

    for (r,c) in pred_pixel:
        plt.scatter(c, r, color="red", label="predicted sources")
    
    cafo_pixel = []
    for lat, lon in cafo_coords:
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        cafo_pixel.append((row,col))

    for (r,c) in cafo_pixel:
        plt.scatter(c, r, color="green", label="cafo sources")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # TODO: Get the predicted model from ERSRR_Model without causing circular imports
    pass
    