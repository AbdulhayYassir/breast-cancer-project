import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- 1. تعريف كلاس الموديل (ضروري جداً لفك ضغط ملف الـ pkl) ---
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

# تصميم الهيدر (تم تصحيح الـ Parameter هنا)
st.markdown("""
    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px">
    <h1 style="color:#2e4053;text-align:center;">🧬 نظام تشخيص سرطان الثدي الذكي</h1>
    <p style="text-align:center;">إدخال يدوي للبيانات أو رفع ملفات شاملة للتحليل الجماعي</p>
    </div>
    """, unsafe_allow_html=True)

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

# أسماء الـ 30 ميزة بالترتيب الصحيح الذي تدرب عليه الموديل
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

if model is None:
    st.error(f"❌ ملف الموديل غير موجود في: {model_path}")
    st.stop()

# --- 4. التبويبات الرئيسية ---
tab1, tab2 = st.tabs(["✍️ فحص حالة واحدة", "📁 تحليل ملف (Batch Mode)"])

# --- التبويب الأول: الإدخال اليدوي ---
with tab1:
    st.info("أدخل قيم الميزات (افتراضياً تم وضع قيم لحالة حميدة):")
    
    # قيم افتراضية آمنة (تمثل حالة حميدة تقريبياً)
    defaults = [12.0, 18.0, 75.0, 450.0, 0.09, 0.08, 0.04, 0.02, 0.17, 0.06] * 3 
    
    user_inputs = []
    cols = st.columns(3) 
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{name}", value=float(defaults[i]), format="%.4f", key=f"manual_{i}")
            user_inputs.append(val)

    if st.button('إجراء التشخيص اليدوي 🔎'):
        features = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        st.divider()
        if prediction == 0:
            st.error("### النتيجة: ورم خبيث (Malignant) ⚠️")
        else:
            st.success("### النتيجة: ورم حميد (Benign) ✅")
            st.balloons()

# --- التبويب الثاني: رفع الملف ---
with tab2:
    st.subheader("تحليل عينات متعددة")
    st.markdown("""
    ارفع ملف CSV يحتوي على الأعمدة الـ 30 بالإضافة لعمود اختياري باسم **Name**.
    """)
    
    uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # التحقق من وجود الأعمدة
            missing_cols = [c for c in feature_names if c not in df.columns]
            
            if not missing_cols:
                # ترتيب الأعمدة وضمان أن البيانات أرقام
                X_batch = df[feature_names].values
                preds = model.predict(X_batch)
                
                # بناء جدول النتائج
                res_df = pd.DataFrame({
                    'الاسم': df['Name'] if 'Name' in df.columns else "مريض مجهول",
                    'التشخيص النهائي': ["خبيث ⚠️" if p == 0 else "حميد ✅" for p in preds]
                })
                
                # دمج النتائج مع البيانات الأصلية للعرض
                final_output = pd.concat([res_df, df[feature_names]], axis=1)
                
                st.success(f"✅ تم تحليل {len(df)} حالة بنجاح!")
                st.dataframe(final_output, use_container_width=True)
                
                # زر التحميل
                csv_file = final_output.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 تحميل تقرير النتائج CSV", csv_file, "diagnosis_report.csv", "text/csv")
            else:
                st.error(f"❌ الملف ينقصه الأعمدة التالية: {', '.join(missing_cols)}")
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف: {e}")