import streamlit as st
import joblib
import numpy as np
import os

# إعداد الصفحة
st.set_page_config(page_title="نظام التشخيص الذكي", page_icon="🩺", layout="wide")

st.title('🩺 نظام تحليل بيانات أورام الثدي')
st.write("يرجى إدخال القياسات الناتجة عن الفحص المجهري بدقة لضمان صحة التوقع.")

# مسار الموديل
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("ملف الموديل غير موجود!")
    st.stop()

# تنظيم المدخلات في أعمدة لشكل أجمل
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input('Mean Radius (نصف القطر)', value=14.0, help="متوسط المسافة من المركز إلى النقاط على المحيط")
    perimeter = st.number_input('Mean Perimeter (المحيط)', value=92.0)
    area = st.number_input('Mean Area (المساحة)', value=650.0)
    smoothness = st.slider('Smoothness (النعومة)', 0.05, 0.25, 0.10)

with col2:
    texture = st.number_input('Mean Texture (النسيج/التباين)', value=19.0)
    concavity = st.slider('Concavity (التجويف)', 0.0, 0.5, 0.08)
    symmetry = st.slider('Symmetry (التماثل)', 0.1, 0.3, 0.18)
    fractal_dim = st.slider('Fractal Dimension', 0.01, 0.1, 0.06)

# تجهيز مصفوفة الـ 30 ميزة
# بنملاها بمتوسطات عامة الأول وبعدين نحط مدخلات المستخدم في أماكنها الصح
input_data = np.zeros((1, 30))
input_data[0, 0] = radius
input_data[0, 1] = texture
input_data[0, 2] = perimeter
input_data[0, 3] = area
input_data[0, 4] = smoothness
input_data[0, 6] = concavity
input_data[0, 8] = symmetry
input_data[0, 9] = fractal_dim

# تعبئة باقي الميزات (من 10 لـ 29) بقيم مرتبطة بالـ radius عشان الموديل ما يتلخبطش
input_data[0, 10:] = radius * 0.1 

st.divider()

if st.button('إجراء تحليل مخبري 🔎'):
    prediction = model.predict(input_data)
    
    st.subheader("النتيجة التحليلية:")
    
    # بناءً على تجاربنا السابقة: 0 حميد و 1 خبيث
    if prediction[0] == 1:
        st.error("⚠️ مؤشرات الورم: خبيث (Malignant)")
        st.info("الخلايا تظهر خصائص غير منتظمة وأحجام متضخمة.")
    else:
        st.success("✅ مؤشرات الورم: حميد (Benign)")
        st.balloons()
        st.info("الخلايا تظهر خصائص منتظمة وضمن النطاق الطبيعي.")