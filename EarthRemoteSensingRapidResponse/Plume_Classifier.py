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


THRESHOLD_VALUE = 0.15 # Threshold value for binary mask. May adust based on model confience
DISTANCE_THRESHOLD = 1.0 # Distance threshold in kilometers for matching predicted plume source to CAFO locations
FILTER_SIZE = 50 # Minimum size of plume region in pixels to be considered valid

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

def compute_centroid(mask, labels, region):
    return ndimage.center_of_mass(mask, labels, region)

def px_to_coords(row, col, transform):
    x, y = rasterio.transform.xy(transform, row, col)
    return (y, x)

def centroid_to_coords(centroid, transform):
    coords = []

    for r,c in centroid:
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
    plume_centroids = compute_centroid(mask, labels, filtered_regions)

    with rasterio.open(raster_path) as src:
        transform = src.transform
    
    predicted_sources = centroid_to_coords(plume_centroids, transform)

    cafo_sources = load_CAFO(cafo_csv)

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

if __name__ == "__main__":
    pred = np.load("path_to_prediction.npy") # Load the model prediction as a numpy array
    raster_path = "path_to_raster.tif" # Path to the raster image used for prediction
    cafo_csv = "path_to_cafo.csv" # Path to the CAFO database CSV file
    results = evaluate_model(pred, cafo_csv, raster_path)

    print("Evaluation Results")
    print("TP:", results["True Positives"])
    print("FP:", results["False Positives"])
    print("FN:", results["False Negatives"])
    print("Precision:", results["Precision"])
    print("Recall:", results["Recall"])
    print("F1 Score:", results["F1_Score"])
    