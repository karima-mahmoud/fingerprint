import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import streamlit as st

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

    # 6. Morphological Processing (Opening)
    morph_image_open = cv2.morphologyEx(morph_image_close, cv2.MORPH_OPEN, kernel)
    steps.append(("Morphological Opening", morph_image_open))

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

# Streamlit UI
st.title("Fingerprint Processing and Matching")
st.markdown("**Upload a fingerprint image and see the processing steps.**")

uploaded_file = st.file_uploader("Upload Query Fingerprint", type=["jpg", "png", "jpeg"])
dataset_files = st.file_uploader("Upload Dataset Fingerprints (Multiple)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_file is not None and dataset_files:
    query_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    dataset_images = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) for f in dataset_files]

    steps = process_image(query_image)

    st.subheader("Processing Steps")
    cols = st.columns(len(steps))

    for col, (step_name, step_image) in zip(cols, steps):
        with col:
            st.image(step_image, caption=step_name, use_column_width=True)
            st.markdown(f"**{step_name}**", unsafe_allow_html=True)

    query_preprocessed = steps[-1][1]
    dataset_preprocessed = [process_image(img)[-1][1] for img in dataset_images]

    match_found, match_index = match_fingerprint(query_preprocessed, dataset_preprocessed)

    st.subheader("Match Results")
    if match_found:
        st.success(f"Match found with dataset image index: {match_index + 1}")
    else:
        st.error("No match found.")
