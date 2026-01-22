import streamlit as st
import joblib
import numpy as np
import os

# --- 1. تعريف الموديل (لازم يكون موجود عشان التحميل ينجح) ---
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

# --- 2. إعداد واجهة المستخدم ---
st.set_page_config(page_title="محلل السرطان الذكي", page_icon="🔬", layout="wide")
st.title('🔬 نظام التشخيص المعتمد على الأشجار القرار')

# --- 3. تحميل الموديل ---
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')

@st.cache_resource
def load_my_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_my_model()

if model is None:
    st.error("❌ لم يتم العثور على ملف الموديل في مجلد models")
    st.stop()

# --- 4. مدخلات البيانات (30 ميزة) ---
st.write("أدخل بيانات الفحص (سيتم استخدام أهم الميزات والباقي سيحسب تلقائياً)")
col1, col2, col3 = st.columns(3)

with col1:
    radius = st.number_input('Mean Radius', value=17.99)
    texture = st.number_input('Mean Texture', value=10.38)
    perimeter = st.number_input('Mean Perimeter', value=122.8)
    area = st.number_input('Mean Area', value=1001.0)

with col2:
    smoothness = st.number_input('Mean Smoothness', value=0.11)
    compactness = st.number_input('Mean Compactness', value=0.27)
    concavity = st.number_input('Mean Concavity', value=0.30)
    concave_points = st.number_input('Mean Concave Points', value=0.14)

with col3:
    worst_radius = st.number_input('Worst Radius', value=25.38)
    worst_perimeter = st.number_input('Worst Perimeter', value=184.6)
    worst_area = st.number_input('Worst Area', value=2019.0)
    worst_concavity = st.number_input('Worst Concavity', value=0.71)

# تجهيز مصفوفة الـ 30 ميزة
features = np.zeros((1, 30))
features[0, 0] = radius
features[0, 1] = texture
features[0, 2] = perimeter
features[0, 3] = area
features[0, 4] = smoothness
features[0, 5] = compactness
features[0, 6] = concavity
features[0, 7] = concave_points
features[0, 20] = worst_radius
features[0, 22] = worst_perimeter
features[0, 23] = worst_area
features[0, 26] = worst_concavity

# --- 5. التوقع والعرض ---
st.divider()
if st.button('تحليل البيانات الآن 🔎'):
    prediction = model.predict(features)
    res = prediction[0]
    
    st.info(f"الرقم الخارج من الموديل: {res}")
    
    # بناءً على الكود بتاعك (غالباً 0 خبيث و 1 حميد)
    if res == 0:
        st.error("⚠️ التشخيص المتوقع: ورم خبيث (Malignant)")
    else:
        st.success("✅ التشخيص المتوقع: ورم حميد (Benign)")
        st.balloons()