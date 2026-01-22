import streamlit as st
import joblib
import numpy as np
import os

# --- 1. تعريف الكلاس (لازم يفضل موجود) ---
class MyDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left, right = tree
        if x[feat_idx] <= threshold:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="فاحص الأورام الذكي", page_icon="🎗️")
st.title('🔬 تشخيص سرطان الثدي (النسخة المتجاوبة)')

# --- 3. تحميل الموديل ---
model = joblib.load('models/my_tree_model.pkl')

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("❌ ملف الموديل غير موجود!")
    st.stop()

# --- 4. مدخلات المستخدم (الأكثر تأثيراً) ---
st.subheader("أدخل البيانات الأساسية للفحص:")
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input('Mean Radius (نصف القطر المتوسط)', value=14.0)
    area = st.number_input('Mean Area (المساحة المتوسطة)', value=650.0)
    concave_points = st.number_input('Mean Concave Points', value=0.05)

with col2:
    w_radius = st.number_input('Worst Radius (أقصى نصف قطر)', value=16.0)
    w_area = st.number_input('Worst Area (أقصى مساحة)', value=880.0)
    w_perimeter = st.number_input('Worst Perimeter (أقصى محيط)', value=100.0)

# --- 5. منطق التوقع (Logic) ---
if st.button('إجراء التحليل 🔎'):
    # مصفوفة تحتوي على القيم المتوسطة للداتا سيت (عشان الموديل ما يتلخبطش بالأصفار)
    # دي قيم الـ Mean لكل الـ 30 ميزة بالترتيب
    input_features = np.array([
        14.12, 19.28, 91.96, 654.8, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062, # Mean
        0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003, # SE
        16.26, 25.67, 107.2, 880.5, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083  # Worst
    ]).reshape(1, -1)

    # تحديث القيم بناءً على مدخلات المستخدم
    input_features[0, 0] = radius
    input_features[0, 3] = area
    input_features[0, 7] = concave_points
    input_features[0, 20] = w_radius
    input_features[0, 22] = w_perimeter
    input_features[0, 23] = w_area

    # التوقع
    prediction = model.predict(input_features)[0]

    # عرض النتيجة
    st.divider()
    st.write(f"**الرقم الخارج من الموديل (Class):** `{prediction}`")
    
    if prediction == 0:
        st.error("### النتيجة المتوقعة: ورم خبيث (Malignant) ⚠️")
        st.write("الموديل لاحظ خصائص تشير إلى نمو غير منتظم.")
    else:
        st.success("### النتيجة المتوقعة: ورم حميد (Benign) ✅")
        st.write("الموديل يشير إلى أن الخصائص ضمن الحدود الطبيعية.")
        st.balloons()