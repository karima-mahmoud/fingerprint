import cv2
import numpy as np

def process_image(image):
    steps = []

    # 1. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("Grayscale Image", gray_image))

    # 2. Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    steps.append(("Blurred Image", blurred_image))

    # 3. Enhance Contrast using Histogram Equalization
    enhanced_image = cv2.equalizeHist(blurred_image)
    steps.append(("Contrast Enhanced Image", enhanced_image))

    # 4. Apply Edge Detection (Canny)
    edges = cv2.Canny(enhanced_image, 50, 150)
    steps.append(("Edge Detection", edges))

    # 5. Morphological Processing (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    steps.append(("Morphological Processing", morph_image))

    # 6. Thinning the ridges
    thinning_image = cv2.ximgproc.thinning(morph_image)
    steps.append(("Thinned Image", thinning_image))

    return steps

def extract_features(image):
    """Extracts features using ORB and returns keypoints and descriptors."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_fingerprint(query_image, dataset_images):
    """Matches a query fingerprint with a dataset of fingerprints."""
    # Extract features from the query image
    keypoints_query, descriptors_query = extract_features(query_image)
    if descriptors_query is None:
        return False, -1  # No features found in query image

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match_idx = -1
    max_good_matches = 0

    for idx, dataset_image in enumerate(dataset_images):
        keypoints_dataset, descriptors_dataset = extract_features(dataset_image)
        if descriptors_dataset is None:
            continue  # Skip if no features found in the dataset image

        matches = bf.match(descriptors_query, descriptors_dataset)
        good_matches = [m for m in matches if m.distance < 50]  # Distance threshold

        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_idx = idx

    # Return the result if sufficient matches are found
    if max_good_matches > 10:  # Minimum threshold for a match
        return True, best_match_idx

    return False, -1

