import streamlit as st
import joblib
import numpy as np
import os

# إعداد واجهة التطبيق
st.set_page_config(page_title="تشخيص سرطان الثدي", page_icon="🩺")

st.title('تشخيص سرطان الثدي باستخدام الذكاء الاصطناعي 🩺')
st.write("أدخل بيانات الفحص السريري للتنبؤ بحالة الورم (حميد/خبيث)")

# تحديد مسار الموديل
model_path = os.path.join(os.getcwd(), 'models', 'breast_cancer_model.pkl')

# تحميل الموديل
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"خطأ: ملف الموديل غير موجود في المسار {model_path}")
    st.stop()

# إنشاء أعمدة للمدخلات
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input('Mean Radius (نصف القطر)', value=14.0)
    texture = st.number_input('Mean Texture (النسيج)', value=19.0)

with col2:
    perimeter = st.number_input('Mean Perimeter (المحيط)', value=92.0)
    area = st.number_input('Mean Area (المساحة)', value=650.0)

# تحضير البيانات للموديل
# الحل الجذري: ملء الـ 30 ميزة بناءً على حجم الـ Radius لضمان استجابة الشجرة
input_data = np.full((1, 30), radius) 
input_data[0, 0:4] = [radius, texture, perimeter, area]

st.divider()

# زر التوقع
if st.button('تحليل النتيجة 🔎'):
    prediction = model.predict(input_data)
    
    # عرض النتيجة
    if prediction[0] == 0:
        st.error("النتيجة المتوقعة: ورم خبيث (Malignant) ⚠️")
        st.write("يُنصح بمراجعة الطبيب المختص فوراً.")
    else:
        st.success("النتيجة المتوقعة: ورم حميد (Benign) ✅")
        st.write("البيانات تشير إلى أن الورم غير مقلق.")

# جزء إضافي للتأكد من البيانات (اختياري)
with st.expander("إحصائيات البيانات المرسلة"):
    st.write(f"المصفوفة المرسلة للموديل (أول 5 قيم): {input_data[0, :5]}")