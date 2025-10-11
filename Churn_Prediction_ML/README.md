# Machine Learning Coursework — Customer Churn Prediction

## 🎯 Objective
This project develops a **customer churn prediction system** using SQL for feature engineering and Python-based machine learning models.  
Built entirely in **Databricks**, it focuses on scalable data preparation, temporal windowing, and robust model validation for predicting customer churn within a 28-day horizon.

---

## ⚙️ Methodology

### **1. Data Windowing**
- Implemented a **14-week tumbling window** to preserve temporal integrity and prevent data leakage.

### **2. Feature Engineering**
- Created behavioral features including visit frequency, spend, category diversity, and recency metrics.
- Added lag-based deltas such as `delta_visits`, `delta_spend`, and `avg_basket_value`.

### **3. Modeling Pipeline**
- Baseline **Logistic Regression** for initial benchmarking.
- Feature selection via **Permutation Importance** to remove weak predictors.
- Model progression: **Logistic Regression → Random Forest → Optimized XGBoost**.
- Hyperparameter tuning using manual grid search to maximize ROC-AUC.

### **4. Evaluation**
- Evaluated models using **Balanced Accuracy** and **ROC-AUC** to handle class imbalance.
- Visualized learning curves to assess overfitting and generalization.

---

## 📈 Key Results
- **Optimized XGBoost** delivered the best predictive performance with stable validation/test results.
- Generated interpretable feature importance plots and churn probability distributions.
- Delivered a reproducible end-to-end churn prediction workflow from SQL to model evaluation.

---

## 🧰 Tools & Technologies
- **Databricks SQL**, **PySpark**, **Pandas**, **Scikit-learn**, **XGBoost**, **Matplotlib**

---

## 📎 Downloads
- **Code (SQL):** [`Machine Learning Coursework.sql`](./Machine%20Learning%20Coursework.sql)
- **Report (Word Document):** [`Machine Learning Coursework Report.docx`](./Machine%20Learning%20Coursework%20Report.docx)

---

