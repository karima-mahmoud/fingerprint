import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert

def process_image(image):
    steps = []

    # 1. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("Grayscale Image", gray_image))

    # 2. Enhance Contrast using Histogram Equalization
    enhanced_image = cv2.equalizeHist(gray_image)
    steps.append(("Contrast Enhanced Image", enhanced_image))

    # 3. Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    steps.append(("Blurred Image", blurred_image))

    # 4. Apply Edge Detection (Canny)
    edges = cv2.Canny(blurred_image, 50, 150)
    steps.append(("Edge Detection", edges))

    # 5. Morphological Processing (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    steps.append(("Morphological Closing", morph_image_close))

    # 6. (Optional) Morphological Processing (Opening)
    morph_image_open = cv2.morphologyEx(morph_image_close, cv2.MORPH_OPEN, kernel)
    steps.append(("Morphological Opening", morph_image_open))

    # 7. Thinning the ridges (Alternative using scikit-image)
    morph_image_binary = morph_image_close // 255  # Convert to binary (0 or 1)
    thinning_image = skeletonize(invert(morph_image_binary)) * 255
    steps.append(("Thinned Image", thinning_image.astype(np.uint8)))

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

# Example usage
if __name__ == "__main__":
    # Load query image and dataset images
    query_image = cv2.imread("query_fingerprint.jpg")
    dataset_images = [
        cv2.imread("fingerprint1.jpg"),
        cv2.imread("fingerprint2.jpg"),
        cv2.imread("fingerprint3.jpg")
    ]

    # Preprocess query image
    steps = process_image(query_image)
    for step_name, step_image in steps:
        cv2.imshow(step_name, step_image)

    # Convert the last step (thinned image) to the format used for matching
    query_preprocessed = steps[-1][1]

    # Preprocess dataset images
    dataset_preprocessed = [process_image(img)[-1][1] for img in dataset_images]

    # Match the query image with the dataset
    match_found, match_index = match_fingerprint(query_preprocessed, dataset_preprocessed)

    if match_found:
        print(f"Match found with image index: {match_index}")
    else:
        print("No match found.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
