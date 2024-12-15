
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from image_processing import process_fingerprint, compare_fingerprints  # استيراد الدوال من الملف الآخر

# إعداد واجهة المستخدم
st.title("Fingerprint Verification System")
st.markdown("Upload a fingerprint image to verify if it belongs to someone in the company.")

# رفع صورة البصمة المدخلة
uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # تحويل الصورة إلى مصفوفة
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # تحويل الصورة إلى مصفوفة NumPy
    image = np.array(image)
    
    # معالجة الصورة
    processed_image = process_fingerprint(image)
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    
    # مقارنة الصورة مع البصمات المخزنة في قاعدة البيانات
    result, name = compare_fingerprints(processed_image)
    
    # عرض النتيجة
    if result:
        st.success(f"The fingerprint belongs to: {name}")
    else:
        st.error("Fingerprint not found in the database.")
