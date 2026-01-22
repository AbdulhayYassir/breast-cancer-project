import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import plotly.express as px  # مكتبة الرسوم التفاعلية

# --- 1. الموديل (كالعادة) ---
class MyDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple): return tree
        feat_idx, threshold, left, right = tree
        if x[feat_idx] <= threshold: return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)

# --- 2. الإعدادات والتحميل ---
st.set_page_config(page_title="AI Cancer Analyzer", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists('models/my_tree_model.pkl'):
        return joblib.load('models/my_tree_model.pkl')
    return None

model = load_model()
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# --- 3. الواجهة الرسومية ---
st.title("📊 لوحة بيانات تشخيص سرطان الثدي")

tab1, tab2 = st.tabs(["🎯 فحص سريع", "📂 تحليل ملفات ضخمة"])

with tab1:
    # (نفس كود الإدخال اليدوي السابق بدون تغيير)
    st.write("أدخل البيانات يدوياً للحصول على تشخيص فوري.")
    # ... (مختصر هنا للتركيز على الجديد في tab2)

with tab2:
    st.header("📂 معالجة البيانات الجماعية")
    uploaded_file = st.file_uploader("ارفع ملف الـ 100 عينة (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in feature_names):
            # التوقع
            X_batch = df[feature_names].values
            preds = model.predict(X_batch)
            
            # تجهيز النتائج
            df['Diagnosis'] = ["Malignant ⚠️" if p == 0 else "Benign ✅" for p in preds]
            
            # --- قسم الإحصائيات (الجديد) ---
            st.divider()
            col_stats1, col_stats2 = st.columns([1, 2])
            
            counts = df['Diagnosis'].value_counts().reset_index()
            counts.columns = ['Status', 'Count']

            with col_stats1:
                st.subheader("📈 ملخص الحالات")
                fig_pie = px.pie(counts, values='Count', names='Status', 
                                 color='Status', 
                                 color_discrete_map={'Malignant ⚠️':'#ef553b', 'Benign ✅':'#00cc96'},
                                 hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_stats2:
                st.subheader("📊 توزيع النتائج")
                fig_bar = px.bar(counts, x='Status', y='Count', color='Status',
                                 color_discrete_map={'Malignant ⚠️':'#ef553b', 'Benign ✅':'#00cc96'})
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()
            st.subheader("📋 الجدول التفصيلي")
            st.dataframe(df[['Name', 'Diagnosis'] + feature_names], use_container_width=True)
            
            # زر التحميل
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل التقرير الكامل", csv, "Full_Report.csv", "text/csv")
        else:
            st.error("الأعمدة غير متوافقة!")