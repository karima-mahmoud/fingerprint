# Fingerprint Verification System

A fingerprint verification system that uses OpenCV and Streamlit for preprocessing and matching fingerprints. The system provides a step-by-step visualization of fingerprint processing and matching against a dataset.

## Features

* Grayscale Conversion
* Contrast Enhancement
* Gaussian Blur
* Edge Detection (Canny)
* Morphological Processing (Closing and Opening)
* Feature Extraction and Matching using ORB (Oriented FAST and Rotated BRIEF)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/karima-mahmoud/fingerprint-verification.git
   cd fingerprint-verification
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Directory Structure

```
./fingerprint-verification/
│
├── app.py                    # Main application file
├── image_processing.py       # Image processing functions
├── requirements.txt          # Required libraries
└── README.md                 # Project documentation (this file)
```

## Usage

1. **Upload Dataset Images:** Use the sidebar to upload multiple fingerprint images for the dataset.
2. **Upload Query Image:** Use the sidebar to upload a single query fingerprint image.
3. **View Processing Steps:** The app will display each step of the fingerprint processing.
4. **Match Verification:** Click the "Match Query with Dataset" button to check if the query fingerprint matches any in the dataset.

## Processing Steps

* **Grayscale Conversion:** Converts the original image to grayscale.
* **Contrast Enhancement:** Uses histogram equalization to enhance the contrast.
* **Gaussian Blur:** Applies a Gaussian blur to reduce noise.
* **Edge Detection:** Extracts edges using the Canny algorithm.
* **Morphological Processing:** Removes noise and fills gaps using morphological closing and opening.

## Future Improvements

* Add support for other feature detectors like SIFT or SURF.
* Implement a more advanced matching algorithm for higher accuracy.
* Include real-time fingerprint scanning support.

## 🙋‍♂️ Author

**👨‍💻 Karima Mahmoud**  
📫 karimamahmoudsalem1@gmail.com  
🐙 GitHub: https://github.com/karima-mahmoud
