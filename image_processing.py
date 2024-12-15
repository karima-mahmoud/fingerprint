import cv2
import numpy as np

def process_image(image):
    steps = []

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("Grayscale Image", gray_image))

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    steps.append(("Blurred Image", blurred_image))

    # Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)
    steps.append(("Edge Detection", edges))

    # Morphological Processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    steps.append(("Morphological Processing", morph_image))

    return steps

def match_fingerprint(query_image, dataset_images):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    keypoints_query, descriptors_query = orb.detectAndCompute(query_image, None)

    for idx, dataset_image in enumerate(dataset_images):
        keypoints_dataset, descriptors_dataset = orb.detectAndCompute(dataset_image, None)
        matches = bf.match(descriptors_query, descriptors_dataset)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:  # Threshold for a match
            return True, idx

    return False, -1
