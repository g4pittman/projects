# Dissertation Title -- Predicting Post-Volatility Stock Corrections Using Machine Learning

## Objective
This project investigates whether machine learning models can predict the **direction and magnitude of stock price corrections** following periods of extreme volatility.  
Using U.S. equity data, macroeconomic indicators, and firm-level fundamentals, the study evaluates whether systematic patterns precede or follow volatility shocks at the firm level.

---

## 📎 Downloads
- **Code (.py):** [`Final Diss Data copy.py`](./Final%20Diss%20Data%20copy.py)
- **Dissertation (PDF):** [`Dissertation copy.pdf`](./Dissertation%20copy.pdf)



---

## Data Sources -- S&P Cap IQ Data Query
1. **Daily Market Data** — Stock price, volume, and market capitalization from `daily_master_final.csv`.  
2. **Firm Fundamentals** — Quarterly accounting data (book value, net income) from `quarterly_raw.csv`.  
3. **Company Metadata** — Credit ratings, dividend information, IPO data from `company_metadata.csv`.  
4. **Macroeconomic Indicator** — U.S. Treasury yield spread (`T10Y2Y.csv`) to capture yield curve inversions.  

After cleaning and merging, the data form a unified **daily-frequency panel** of all stocks with both static and time-varying features.

---

## Methodology

### **1. Data Cleaning and Integration**
- Converted dates and standardized formats across all datasets.  
- Encoded **credit ratings** into ordinal ranks (`credit_rating_rank`).  
- Forward-filled missing fundamentals using only past data to prevent leakage.  
- Merged all datasets chronologically via an *as-of merge*, ensuring each trading day included the most recent firm and macro data.

### **2. Feature Engineering**
- **Lagged prices & returns:** 1, 5, 10, 20-day lags  
- **Technical indicators:** Moving averages, RSI (14-day), volume trends, and momentum metrics  
- **Market-wide volatility:** Daily average absolute return and rolling volatility (1–20-day)  
- **Macro features:** Lagged and delta values of the yield spread (`T10Y2Y`)  
- **Fundamental features:** Most recent net income and book value equity  
- **Targets:** Forward 1-day, 5-day, and 20-day returns following each volatility shock

### **3. Volatility Shock Identification**
- Defined shocks as **±2 standard deviations** from each firm’s mean daily return.  
- Isolated non-redundant shocks using streak logic (ensuring one observation per event).  
- Labeled shocks as **positive or negative** based on daily return polarity.

### **4. Modeling Approach**
Implemented multiple supervised regression models:
- **Baseline Mean Predictor** — Benchmark for RMSE/MAE  
- **Random Forest Regressor**  
- **XGBoost Regressor (tuned)**  
- **CatBoost Regressor**

Evaluated using:
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² Score  
- Directional Accuracy (correct sign of predicted return)

### **5. Hyperparameter Optimization**
- Used **GroupKFold CV (by entity_ID)** to prevent leakage across firms.  
- Applied **RandomizedSearchCV** and manual parameter tuning for robustness.

### **6. Feature Evaluation**
Derived feature importance using:
- CatBoost built-in importance  
- XGBoost gain-based importance  
- Permutation importance  
- SHAP value analysis  

**Top Predictors:**
- Recent returns (t-1, t-5)  
- Market volatility measures  
- RSI  
- Yield curve movement  
- Net income and book value equity

---

## Key Outputs and Insights
- Created a **master modeling dataset** with 40+ engineered features per stock-day.  
- Identified and visualized **volatility events**, **shock polarity**, and **market stress periods**.  
- Models achieved above-baseline accuracy and strong **directional predictability**.  
- SHAP analysis revealed that **market volatility**, **RSI**, and **momentum** were most influential.

---

## Deliverables
- `master_df` — Full merged and cleaned dataset  
- `vol_df_global` — Filtered volatility-event dataset  
- Model performance metrics and feature importance plots  
- Visual analyses:
  - Volatility shock distributions  
  - Rolling volatility plots  
  - SHAP summaries and waterfalls  

---

## 📊 Visuals

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Dissertation_Stock_Prediction_Model/Final%20Permutation%20Importance%2020d.png?raw=true" width="650"/>
</p>

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Dissertation_Stock_Prediction_Model/Net%20Shock%20Score%20Over%20Time%20copy.png?raw=true" width="650"/>
</p>

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Dissertation_Stock_Prediction_Model/SHAP%20Value%20Feature%Importance%20copy.png?raw=true" width="650"/>
</p>


<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Dissertation_Stock_Prediction_Model/Stacked%20Area%20of%20Pos%20Neg%20Shocks%20Over%20Time%20copy.png?raw=true" width="650"/>
</p>

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Dissertation_Stock_Prediction_Model/Yield%20Curve%20Regime%20Across%20Dataset.png?raw=true" width="650"/>
</p>


---


## Tools and Technologies
- **Python** — Pandas, NumPy, Matplotlib, Seaborn  
- **ML Frameworks** — XGBoost, CatBoost, scikit-learn, SHAP  
- **Feature Engineering**  
- **Cross-Validation** — GroupKFold  
- **Environment** — Google Colab, S&P Cap IQ 

---

## Conclusion
Machine learning models, especially gradient boosting methods like **XGBoost** and **CatBoost**, can partially anticipate the **direction and extent of short-term corrections** following volatility shocks.  
This framework enhances understanding of **post-shock market dynamics** and provides a scalable structure for volatility-event modeling in financial time-series data.

