# Breast Cancer Classification (Custom Decision Tree) 🩺

مشروع لتصنيف سرطان الثدي باستخدام خوارزمية شجرة القرار (Decision Tree) مبنية من الصفر.

## 📊 النتائج (Performance)
- **منطق الموديل:** تم استخدام **Information Gain** و **Entropy**.
- **دقة التدريب (Train Accuracy):** 100%
- **دقة الاختبار (Test Accuracy):** 92.98%
- **حالة الموديل:** أداء ممتاز مع نسبة Overfitting ضئيلة جداً (7%).

## 🏗️ هيكل المشروع (Structure)
- `src/components/model.py`: الكود الأساسي للموديل (Custom Class).
- `src/components/model_trainer.py`: المسؤول عن تدريب وحفظ الموديل.
- `src/components/model_evaluation.py`: ملف التقييم وحساب الأوفر فيتنج.

## 🚀 التشغيل (Quick Start)
1. **تحميل المكتبات:**
   ```bash
   pip install -r requirements.txt
   ```
2. **تدريب الموديل:**
   ```bash
   python3 -m src.components.model_trainer
   ```
3. **التقييم:**
   ```bash
   python3 -m src.components.model_evaluation
   ```
