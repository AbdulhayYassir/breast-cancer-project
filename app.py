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
default_means = [
    14.12, 19.28, 91.96, 654.8, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062,
    0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003,
    16.26, 25.67, 107.2, 880.5, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083
]

iinput_data = np.full((1, 30), radius) # بنملا الـ 30 ميزة بنفس رقم الـ radius كبداية
input_data[0, 0:4] = [radius, texture, perimeter, area]

if st.button("تحليل النتيجة"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("النتيجة: خبيث (Malignant) ⚠️")
    else:
        st.success("النتيجة: حميد (Benign) ✅")

st.info("ملاحظة: هذا النموذج لأغراض تعليمية فقط.")
