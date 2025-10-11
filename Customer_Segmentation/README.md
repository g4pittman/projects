# Customer Segmentation Analysis — Retail Analytics Case Study

## 🎯 Objective
Conduct a **customer segmentation analysis** for a national convenience store chain to identify **five to seven distinct customer groups** based on purchasing behavior.  
The goal was to help the company understand shopper behavior, optimize retention strategies, and design targeted marketing campaigns.

---

## ⚙️ Methodology

### **1. Data Preparation**
- Dataset: 3,000 customers over a six-month period.
- Transactional data at multiple levels: **customer**, **basket**, **category**, and **line-item**.
- Cleaned and standardized data to handle missing or errant values.
- Engineered behavioral features focused on **Recency, Frequency, and Monetary (RFM)** metrics:contentReference[oaicite:0]{index=0}.

### **2. Feature Engineering**
Created comprehensive feature sets across four dimensions:
- **Category-Level:** Aggregated 18 product types into 3 major groups — *fresh foods, processed foods, and non-food items* — to preserve data integrity while reducing noise.
- **Basket-Level:** Measured total spend, average item count, and category specialization per basket.
- **Temporal:** Captured time-of-day and weekend vs. weekday shopping trends.
- **Transactional:** Total and average basket spend, plus concentration of top product categories.

### **3. Dimensionality Reduction & Clustering**
- Applied **Non-negative Matrix Factorization (NMF)** and **Principal Component Analysis (PCA)** for dimensionality reduction.  
- NMF performed best, producing **five well-separated clusters** using **K-Means Clustering** with optimized silhouette and elbow validation.
- Normalized features using **Min-Max Scaling** to balance scale differences and maintain interpretability.

---

## 📈 Key Insights
**Five distinct customer groups** were identified:
1. **High Value Shoppers** — frequent visitors with high total spend; ideal for loyalty programs.  
2. **Quick Morning Stop Shoppers** — low-spend, high morning visit rates; target with after-work promotions.  
3. **Morning Mission Bulk Buyers** — prefer early bulk purchases; cross-sell household and non-food items.  
4. **One-Item Random Stop Shoppers** — infrequent visitors with low basket size; incentivize with discounts.  
5. **Moderate Shoppers** — steady and loyal; maintain engagement through rewards programs:contentReference[oaicite:1]{index=1}.

---

## 💼 Business Recommendations
- Prioritize **retention of High Value Shoppers** via premium loyalty programs.
- Target **One-Item Shoppers** with discounts to convert them into regulars.
- Use behavioral clusters to tailor campaigns (e.g., weekend vs. weekday offers).
- Extend segmentation insights to store layout optimization and inventory planning.

---

## 🧰 Tools & Techniques
- **Python (Pandas, Matplotlib, scikit-learn)**  
- **K-Means Clustering**, **PCA**, **NMF**, **Silhouette Analysis**  
- **SQL** for data preparation and summarization  

---

## 📎 Downloads
- **Code (Python Script):** [`20595117_BUSI4370_2024-25.py`](./20595117_BUSI4370_2024-25.py)
- **Report (Word Document):** [`20595117_BUSI4370_2024-25.docx`](./20595117_BUSI4370_2024-25.docx)

---

