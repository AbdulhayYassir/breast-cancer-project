import streamlit as st
import joblib
import numpy as np
import os

# إعداد الصفحة وتوسيع العرض
st.set_page_config(page_title="المحلل الشامل لأورام الثدي", page_icon="🔬", layout="wide")

st.title('🔬 المحلل الشامل لبيانات أورام الثدي (30 ميزة)')
st.write("قم بإدخال كافة المؤشرات الحيوية للحصول على أدق نتيجة ممكنة.")

# تحميل الموديل
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("خطأ: لم يتم العثور على ملف الموديل!")
    st.stop()

# إنشاء تبويبات لتنظيم الـ 30 ميزة
tab1, tab2, tab3 = st.tabs(["📊 قيم المتوسط (Mean)", "📉 قيم الخطأ (SE)", "⚠️ قيم الأسوأ (Worst)"])

features = []

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        features.append(st.number_input('Mean Radius', value=17.99))
        features.append(st.number_input('Mean Texture', value=10.38))
        features.append(st.number_input('Mean Perimeter', value=122.8))
        features.append(st.number_input('Mean Area', value=1001.0))
        features.append(st.number_input('Mean Smoothness', value=0.118))
    with col2:
        features.append(st.number_input('Mean Compactness', value=0.277))
        features.append(st.number_input('Mean Concavity', value=0.300))
        features.append(st.number_input('Mean Concave Points', value=0.147))
        features.append(st.number_input('Mean Symmetry', value=0.241))
        features.append(st.number_input('Mean Fractal Dimension', value=0.078))

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        for i in range(5): # أول 5 فيتشرز في الـ Error
            features.append(st.number_input(f'Error Feature {i+1}', value=0.5, key=f"err_{i}"))
    with col4:
        for i in range(5, 10): # ثاني 5 فيتشرز في الـ Error
            features.append(st.number_input(f'Error Feature {i+1}', value=0.03, key=f"err_{i}"))

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        features.append(st.number_input('Worst Radius', value=25.38))
        features.append(st.number_input('Worst Texture', value=17.33))
        features.append(st.number_input('Worst Perimeter', value=184.6))
        features.append(st.number_input('Worst Area', value=2019.0))
        features.append(st.number_input('Worst Smoothness', value=0.162))
    with col6:
        features.append(st.number_input('Worst Compactness', value=0.665))
        features.append(st.number_input('Worst Concavity', value=0.711))
        features.append(st.number_input('Worst Concave Points', value=0.265))
        features.append(st.number_input('Worst Symmetry', value=0.460))
        features.append(st.number_input('Worst Fractal Dimension', value=0.118))

# تحويل القائمة لمصفوفة numpy جاهزة للموديل
input_data = np.array(features).reshape(1, -1)

st.divider()

if st.button('إجراء التحليل النهائي 🔎'):
    prediction = model.predict(input_data)
    
    st.subheader("النتيجة:")
    if prediction[0] == 1:
        st.error("النتيجة المتوقعة: ورم خبيث (Malignant) ⚠️")
    else:
        st.success("النتيجة المتوقعة: ورم حميد (Benign) ✅")
        st.balloons()