# Predictive Marketing Model — N/LAB Platinum Deposit Campaign

## 🎯 Objective
Develop a **machine learning model** to optimize outbound marketing calls for N/LAB Banking’s *Platinum Deposit* product.  
The goal was to **reduce unproductive calls** and **maximize sales conversion** by predicting which customers were most likely to purchase the financial product.

---

## ⚙️ Methodology

### **1. Exploratory Data Analysis**
- Analyzed 4,000 customer observations with a 21% positive response baseline.
- Visualized relationships through histograms, box plots, and correlation matrices.
- Identified top predictive variables:
  - **Poutcome (previous campaign outcome)**
  - **Pdays (days since last contact)**
  - **Previous (number of prior contacts)**
  - **Age**, **Job**, and **Loan** status:contentReference[oaicite:0]{index=0}.

### **2. Feature Insights**
- **Poutcome:** Customers with a previous “success” showed an 83% buy rate — a 62% increase over baseline.
- **Pdays:** Optimal contact window = **50–200 days** since last contact.
- **Demographics:** Highest buy rates among **students (41%)** and **retirees (37%)**, lowest among blue-collar workers.
- **Loans:** Absence of housing/personal loans correlated with higher purchase likelihood.

### **3. Modeling Process**
Three supervised learning algorithms were tested:
- **Logistic Regression** – strong interpretability, limited performance due to data non-linearity.
- **Decision Tree** – provided valuable insights into customer subgroups but prone to overfitting.
- **Random Forest** – achieved the best performance with **86% precision** and **20% recall**, excelling in negative response prediction.

### **4. Model Implementation**
- Deployed a **Random Forest classifier** with tunable precision–recall trade-offs.
- Created a **user-friendly model interface** allowing:
  - CSV upload and retraining on new data.
  - Adjustable probability thresholds for call prioritization.
  - Direct export of high-probability customer lists.

---

## 📈 Key Business Recommendations
- Prioritize **customers previously contacted successfully**.
- Focus on **retired individuals** and **students** without outstanding loans.
- Re-contact customers **50–200 days** after last engagement.
- Retrain the model regularly as more data accumulates.
- Develop a separate demographic-only model for **new customer acquisition**.

---

## 🔍 Future Opportunities
- Enhanced segmentation of high-performing groups.
- Expansion of training data to improve recall.
- Exploration of additional behavioral and financial indicators.

---

## 🧰 Tools & Technologies
- **Python (Jupyter Notebook)**, **SQL**, **Pandas**, **Scikit-learn**, **Matplotlib**
- **Decision Tree**, **Random Forest**, **Logistic Regression**

---

## 📊 Visuals

<p align="center">
  <h3><strong>📈 Campaign Conversion Funnel</strong></h3>
  <img src="https://github.com/g4pittman/projects/blob/main/Customer_Conversion_Prediction_Marketing_Campaign/download-1.png?raw=true" width="600"/>
  <br>
  <em>Visualizing the end-to-end marketing conversion funnel, from impressions to final customer acquisition.</em>
</p>

<p align="center">
  <h3><strong>🎯 Channel Performance Comparison</strong></h3>
  <img src="https://github.com/g4pittman/projects/blob/main/Customer_Conversion_Prediction_Marketing_Campaign/download-2.png?raw=true" width="600"/>
  <br>
  <em>Comparing campaign success rates across digital channels to identify the most effective acquisition sources.</em>
</p>

<p align="center">
  <h3><strong>💰 ROI and Cost Efficiency Analysis</strong></h3>
  <img src="https://github.com/g4pittman/projects/blob/main/Customer_Conversion_Prediction_Marketing_Campaign/download-3.png?raw=true" width="600"/>
  <br>
  <em>Assessing marketing return on investment (ROI) and cost-per-acquisition metrics for performance optimization.</em>
</p>

---

## 📎 Downloads
- **Code (Jupyter Notebook):** [`FBA_Final_Coursework.ipynb`](./FBA_Final_Coursework.ipynb)
- **Final Model (Jupyter Notebook):** [`FBA_Final_Model.ipynb`](./FBA_Final_Model.ipynb)
- **Report (Word Document):** [`Project V.2.docx`](./Project%20V.2.docx)

---

