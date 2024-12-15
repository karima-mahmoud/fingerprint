import streamlit as st
from PIL import Image
import cv2
import numpy as np
from image_processing import process_image, match_fingerprint

# Global dataset
dataset = []

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

st.title("Fingerprint Verification System")

# Step 1: Upload dataset
st.sidebar.header("Dataset Management")
uploaded_dataset = st.sidebar.file_uploader("Upload Dataset Images", accept_multiple_files=True)
if uploaded_dataset:
    dataset = [load_image(file) for file in uploaded_dataset]
    st.sidebar.success(f"Uploaded {len(dataset)} images to the dataset.")

# Step 2: Upload query image
st.sidebar.header("Query Image")
query_file = st.sidebar.file_uploader("Upload Query Fingerprint Image")
query_image = load_image(query_file) if query_file else None

# Step 3: Process and match
if query_image is not None:
    st.subheader("Processing Steps")
    steps = process_image(query_image)

    for title, img in steps:
        st.image(img, caption=title, use_column_width=True, channels="GRAY")

    if st.button("Match Query with Dataset"):
        match, index = match_fingerprint(cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY), dataset)

        if match:
            st.success(f"Fingerprint matched with dataset image #{index + 1}.")
        else:
            st.error("No match found in the dataset.")

