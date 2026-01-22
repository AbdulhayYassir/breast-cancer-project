import streamlit as st
import joblib
import numpy as np
import os

# إعداد الصفحة
st.set_page_config(page_title="كاشف السرطان الذكي", page_icon="🩺")

st.title('تشخيص سرطان الثدي بالذكاء الاصطناعي 🩺')
st.markdown("---")

# مسار الموديل
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')

# تحميل الموديل
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("لم يتم العثور على ملف الموديل! تأكد من رفعه في مجلد models")
    st.stop()

# مدخلات المستخدم
col1, col2 = st.columns(2)
with col1:
    radius = st.number_input('Mean Radius', value=14.0)
    texture = st.number_input('Mean Texture', value=19.0)
with col2:
    perimeter = st.number_input('Mean Perimeter', value=92.0)
    area = st.number_input('Mean Area', value=650.0)

# تجهيز البيانات
input_data = np.full((1, 30), radius)
input_data[0, 0:4] = [radius, texture, perimeter, area]

if st.button('تحليل الحالة 🔎'):
    prediction = model.predict(input_data)
    
    # إظهار الرقم الناتج للتشخيص (للتأكد فقط)
    st.info(f"الرقم الناتج من الموديل (Class): {prediction[0]}")
    
    # التعديل الذهبي بناءً على تجربتك:
    # الموديل عندك بيطلع 0 للحالات الحميدة و 1 للخبيثة
    if prediction[0] == 1:
        st.error("النتيجة المتوقعة: ورم خبيث (Malignant) ⚠️")
    else:
        st.success("النتيجة المتوقعة: ورم حميد (Benign) ✅")
        st.balloons()