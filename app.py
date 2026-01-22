import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- 1. تعريف الكلاس (يجب أن يظل كما هو) ---
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
st.set_page_config(page_title="نظام التشخيص المتكامل", page_icon="🧬", layout="wide")
st.title('🧬 نظام تحليل سرطان الثدي الذكي (إدخال يدوي + رفع ملفات)')

# --- 3. تحميل الموديل ---
model_path = 'models/my_tree_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# أسماء الـ 30 ميزة بالترتيب الصحيح للموديل
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

if model is None:
    st.error("❌ ملف الموديل غير موجود!")
    st.stop()

# --- 4. التبويبات (Tabs) ---
tab1, tab2 = st.tabs(["✍️ فحص حالة واحدة", "📁 رفع ملف عينات (Batch)"])

with tab1:
    st.subheader("أدخل الـ 30 ميزة يدوياً:")
    user_inputs = []
    
    # تقسيم الـ 30 ميزة على 3 أعمدة
    cols = st.columns(3)
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{name}", value=0.0, format="%.4f", key=f"manual_{i}")
            user_inputs.append(val)

    if st.button('تحليل الحالة اليدوية 🔎'):
        features = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        if prediction == 0:
            st.error("### النتيجة: ورم خبيث (Malignant) ⚠️")
        else:
            st.success("### النتيجة: ورم حميد (Benign) ✅")

with tab2:
    st.subheader("رفع ملف بيانات للاختبار")
    st.write("ارفع ملف CSV يحتوي على الأعمدة الـ 30 بالإضافة لعمود 'Name'.")
    
    uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # التأكد من وجود الأعمدة
        if all(col in df.columns for col in feature_names):
            st.write("✅ تم العثور على جميع الميزات المطلوبة.")
            
            # التحليل
            X_batch = df[feature_names].values
            predictions = model.predict(X_batch)
            
            # تجهيز النتائج
            results_df = pd.DataFrame({
                'الاسم': df['Name'] if 'Name' in df.columns else "مجهول",
                'النتيجة الرقمية': predictions,
                'التشخيص النهائي': ["خبيث ⚠️" if p == 0 else "حميد ✅" for p in predictions]
            })
            
            st.divider()
            st.subheader("📋 نتائج التحليل الجماعي:")
            st.dataframe(results_df, use_container_width=True)
            
            # زر لتحميل النتائج
            csv = results_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل النتائج كملف CSV", csv, "diagnosis_results.csv", "text/csv")
        else:
            st.error("❌ الملف المرفوع لا يحتوي على كل الأعمدة الـ 30 المطلوبة!")