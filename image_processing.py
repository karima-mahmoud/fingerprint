

import cv2
import numpy as np

# دالة لمعالجة صورة البصمة
def process_fingerprint(image):
    # تحويل الصورة إلى تدرجات الرمادي
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين الصورة باستخدام Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # تطبيق عتبة لتحويل الصورة إلى صورة ثنائية (Binary)
    _, thresholded_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)
    
    return thresholded_image

# دالة للمقارنة مع البصمات المخزنة
def compare_fingerprints(processed_image):
    # هنا يمكنك استخدام أي خوارزمية للمقارنة مثل SIFT أو SURF أو ببساطة مقارنة الصور
    # سنستخدم صورة مخزنة كمثال (من قاعدة بيانات افتراضية)
    
    # تحميل صورة من قاعدة البيانات (مثال)
    stored_image = cv2.imread('dataset/sample_fingerprint.jpg', cv2.IMREAD_GRAYSCALE)
    
    # معالجة الصورة المخزنة بنفس الطريقة
    processed_stored_image = process_fingerprint(stored_image)
    
    # مقارنة الصور باستخدام المسافة الإقليدية أو أسلوب آخر
    distance = np.linalg.norm(processed_image - processed_stored_image)
    
    # تحديد حد للمقارنة
    threshold = 1000  # يمكنك تعديل هذا حسب الدقة المطلوبة
    
    if distance < threshold:
        return True, "John Doe"  # اسم الشخص إذا كانت البصمة متطابقة
    else:
        return False, ""
