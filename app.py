import streamlit as st
import joblib
import numpy as np
import os

# 1. إعداد الصفحة
st.set_page_config(page_title="المحلل الشامل - النسخة النهائية", page_icon="🔬", layout="wide")

st.title('🔬 نظام التشخيص المتكامل (30 ميزة)')
st.write("أدخل البيانات كاملة أدناه. هذا الكود مصمم لكشف لغز التصنيف.")

# 2. تحميل الموديل
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("خطأ: ملف الموديل breast_cancer_model.pkl غير موجود في مجلد models!")
    st.stop()

# 3. واجهة المستخدم (30 ميزة مقسمة)
tab1, tab2, tab3 = st.tabs(["📊 المتوسط (Mean)", "📉 الخطأ (SE)", "⚠️ الأسوأ (Worst)"])

all_features = []

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input('Mean Radius', value=17.99)
        f2 = st.number_input('Mean Texture', value=10.38)
        f3 = st.number_input('Mean Perimeter', value=122.8)
        f4 = st.number_input('Mean Area', value=1001.0)
        f5 = st.number_input('Mean Smoothness', value=0.118)
    with col2:
        f6 = st.number_input('Mean Compactness', value=0.277)
        f7 = st.number_input('Mean Concavity', value=0.300)
        f8 = st.number_input('Mean Concave Points', value=0.147)
        f9 = st.number_input('Mean Symmetry', value=0.241)
        f10 = st.number_input('Mean Fractal Dimension', value=0.078)
    all_features.extend([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

with tab2:
    cols = st.columns(2)
    for i in range(10):
        with cols[i % 2]:
            val = st.number_input(f'Error Feature {i+1}', value=0.5, key=f"se_{i}")
            all_features.append(val)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.number_input('Worst Radius', value=25.38)
        w2 = st.number_input('Worst Texture', value=17.33)
        w3 = st.number_input('Worst Perimeter', value=184.6)
        w4 = st.number_input('Worst Area', value=2019.0)
        w5 = st.number_input('Worst Smoothness', value=0.162)
    with col2:
        w6 = st.number_input('Worst Compactness', value=0.665)
        w7 = st.number_input('Worst Concavity', value=0.711)
        w8 = st.number_input('Worst Concave Points', value=0.265)
        w9 = st.number_input('Worst Symmetry', value=0.460)
        w10 = st.number_input('Worst Fractal Dimension', value=0.118)
    all_features.extend([w1, w2, w3, w4, w5, w6, w7, w8, w9, w10])

# 4. التوقع والتحليل
st.divider()
if st.button('إجراء التحليل النهائي 🔎'):
    input_data = np.array(all_features).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # محاولة عرض الاحتمالات لو الموديل بيدعمها
    try:
        probs = model.predict_proba(input_data)
        st.write(f"📊 احتمالات التصنيف (Probabilities): {probs[0]}")
    except:
        pass

    st.info(f"🔢 الرقم الخارج من الموديل (Class): {prediction[0]}")

    # التعديل بناءً على سلوك الموديل عندك:
    # لو الموديل بيطلع 0 مع الأرقام الكبيرة (الخبيثة)، هنخلي الـ 0 هي الـ Malignant
    if prediction[0] == 0:
        st.error("⚠️ التشخيص: ورم خبيث (Malignant)")
    else:
        st.success("✅ التشخيص: ورم حميد (Benign)")
        st.balloons()