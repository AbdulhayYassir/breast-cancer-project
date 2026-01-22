import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import plotly.express as px

# --- 1. هيكل الموديل ---
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

# --- 2. إعدادات وتحميل ---
st.set_page_config(page_title="Pro Cancer AI Analyzer", page_icon="🧬", layout="wide")

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

# --- 3. تصميم الواجهة ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 نظام التحليل التشخيصي المتطور")
st.markdown("---")

tab1, tab2 = st.tabs(["🎯 فحص حالة فردية", "📊 مستودع البيانات الضخمة (2000+ حالة)"])

# --- التبويب الأول: يدوي ---
with tab1:
    col_input, col_res = st.columns([2, 1])
    with col_input:
        st.subheader("📝 إدخال بيانات المريض")
        defaults = [12.0, 18.0, 75.0, 450.0, 0.09, 0.08, 0.04, 0.02, 0.17, 0.06] * 3
        user_inputs = []
        c = st.columns(3)
        for i, name in enumerate(feature_names):
            with c[i % 3]:
                val = st.number_input(f"{name}", value=float(defaults[i]), key=f"m_{i}")
                user_inputs.append(val)
    
    with col_res:
        st.subheader("🔍 نتيجة التحليل")
        if st.button("تحليل الحالة الآن", use_container_width=True):
            pred = model.predict(np.array(user_inputs).reshape(1, -1))[0]
            if pred == 0:
                st.error("### النتيجة: ورم خبيث ⚠️")
                st.progress(100)
            else:
                st.success("### النتيجة: ورم حميد ✅")
                st.balloons()

# --- التبويب الثاني: معالجة الملفات والبحث ---
with tab2:
    uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in feature_names):
            # التوقعات
            X = df[feature_names].values
            preds = model.predict(X)
            df['Diagnosis'] = ["خبيث ⚠️" if p == 0 else "حميد ✅" for p in preds]
            
            # 1. بطاقات الإحصائيات
            m_count = sum(preds == 0)
            b_count = sum(preds == 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("إجمالي المرضى", len(df))
            c2.metric("حالات خبيثة", m_count, delta=f"{m_count/len(df)*100:.1f}%", delta_color="inverse")
            c3.metric("حالات حميدة", b_count, delta=f"{b_count/len(df)*100:.1f}%")
            
            # 2. الرسوم البيانية
            st.divider()
            g1, g2 = st.columns(2)
            fig_pie = px.pie(values=[m_count, b_count], names=['خبيث', 'حميد'], 
                             color=['خبيث', 'حميد'], color_discrete_map={'خبيث':'#ef553b', 'حميد':'#00cc96'},
                             title="نسبة التوزيع العام")
            g1.plotly_chart(fig_pie, use_container_width=True)
            
            fig_scatter = px.scatter(df, x='mean radius', y='mean texture', color='Diagnosis',
                                    title="توزيع المرضى حسب القطر والملمس",
                                    color_discrete_map={'خبيث ⚠️':'#ef553b', 'حميد ✅':'#00cc96'})
            g2.plotly_chart(fig_scatter, use_container_width=True)

            # 3. محرك البحث والفلترة
            st.divider()
            st.subheader("📋 قاعدة بيانات الفحوصات")
            search_col, filter_col = st.columns([2, 1])
            search_term = search_col.text_input("🔍 ابحث باسم المريض...")
            filter_type = filter_col.selectbox("فلترة حسب الحالة", ["الكل", "خبيث ⚠️", "حميد ✅"])
            
            # تطبيق الفلترة
            view_df = df.copy()
            if search_term:
                view_df = view_df[view_df['Name'].str.contains(search_term, case=False, na=False)]
            if filter_type != "الكل":
                view_df = view_df[view_df['Diagnosis'] == filter_type]
            
            st.dataframe(view_df[['Name', 'Diagnosis'] + feature_names], use_container_width=True)
            
            # تحميل النتائج المفلترة
            csv = view_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل هذه النتائج (CSV)", csv, "Filtered_Report.csv")
        else:
            st.error("الملف لا يحتوي على الـ 30 ميزة المطلوبة!")