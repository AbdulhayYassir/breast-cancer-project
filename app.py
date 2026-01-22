import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- 1. تعريف كلاس الموديل (لازم يكون موجود عشان يقرأ ملف pkl صح) ---
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

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🧬", layout="wide")

# تصميم الهيدر
st.markdown("""
    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px">
    <h1 style="color:#2e4053;text-align:center;">🧬 نظام تشخيص سرطان الثدي الذكي</h1>
    <p style="text-align:center;">إدخال يدوي للبيانات أو رفع ملفات شاملة للتحليل الجماعي</p>
    </div>
    """, unsafe_allow_status=True)

# --- 3. تحميل الموديل ---
model_path = 'models/my_tree_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            return None
    return None

model = load_model()

# أسماء الـ 30 ميزة بالترتيب الصحيح
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

if model is None:
    st.error(f"❌ لم يتم العثور على الموديل في المسار: {model_path}. تأكد من رفع الملف!")
    st.stop()

# --- 4. التبويبات ---
tab1, tab2 = st.tabs(["✍️ فحص حالة واحدة", "📁 تحليل ملف (Batch Mode)"])

# --- التبويب الأول: الإدخال اليدوي ---
with tab1:
    st.info("قم بتعديل قيم الميزات الأساسية أدناه للتحليل:")
    
    # سنضع قيم افتراضية تمثل حالة "حميدة" (Safe Start)
    defaults = [12.0, 18.0, 75.0, 450.0, 0.09, 0.08, 0.04, 0.02, 0.17, 0.06] * 3 
    
    user_inputs = []
    cols = st.columns(3) # تقسيم الشاشة لـ 3 أعمدة
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{name}", value=float(defaults[i]), format="%.4f")
            user_inputs.append(val)

    if st.button('إجراء التشخيص اليدوي 🔎'):
        features = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        st.divider()
        if prediction == 0:
            st.error("### النتيجة: ورم خبيث (Malignant) ⚠️")
            st.write("بناءً على المعطيات، الموديل يصنف هذه الحالة كإصابة خبيثة.")
        else:
            st.success("### النتيجة: ورم حميد (Benign) ✅")
            st.balloons()
            st.write("بناءً على المعطيات، الموديل يصنف هذه الحالة كإصابة حميدة.")

# --- التبويب الثاني: رفع ملف ---
with tab2:
    st.subheader("تحليل عينات متعددة من ملف CSV")
    st.write("تأكد أن الملف يحتوي على عمود 'Name' وأعمدة الميزات الـ 30.")
    
    uploaded_file = st.file_uploader("ارفع ملف البيانات هنا", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # التأكد من ترتيب الأعمدة لتطابق الموديل
        try:
            X_batch = df[feature_names].values
            preds = model.predict(X_batch)
            
            # عرض النتائج في جدول
            res_df = pd.DataFrame({
                'الاسم': df['Name'] if 'Name' in df.columns else "مريض مجهول",
                'التشخيص': ["خبيث ⚠️" if p == 0 else "حميد ✅" for p in preds]
            })
            
            st.success("تم الانتهاء من تحليل جميع العينات!")
            st.dataframe(res_df, use_container_width=True)
            
            # خيار تحميل النتائج
            csv_output = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل التقرير النهائي", csv_output, "results.csv", "text/csv")
            
        except KeyError:
            st.error("❌ فشل التحليل: تأكد أن أسماء الأعمدة في ملفك تطابق تماماً أسماء الميزات المطلوبة.")