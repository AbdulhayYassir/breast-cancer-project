import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# إعداد شكل الصفحة
st.set_page_config(page_title="كاشف السرطان", page_icon="🩺")

# تحميل الموديل (تأكد من المسار)
model_path = 'models/breast_cancer_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("الموديل غير موجود! تأكد من تشغيل model_trainer أولاً.")

st.title("تشخيص سرطان الثدي 🩺")
st.markdown("هذا التطبيق يستخدم **Decision Tree** تم بناؤه من الصفر للتوقع.")

# تقسيم الشاشة لخانات إدخال
st.sidebar.header("إدخال البيانات")

# الميزات الأساسية (مثال لأهم ميزات)
radius = st.sidebar.number_input("Mean Radius", value=17.99)
texture = st.sidebar.number_input("Mean Texture", value=10.38)
perimeter = st.sidebar.number_input("Mean Perimeter", value=122.8)
area = st.sidebar.number_input("Mean Area", value=1001.0)

# بقية الـ 30 ميزة هنكملهم بقيم افتراضية عشان الموديل يشتغل
input_data = np.zeros((1, 30))
input_data[0, 0:4] = [radius, texture, perimeter, area]

if st.button("تحليل النتيجة"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("النتيجة: خبيث (Malignant) ⚠️")
    else:
        st.success("النتيجة: حميد (Benign) ✅")

st.info("ملاحظة: هذا النموذج لأغراض تعليمية فقط.")
