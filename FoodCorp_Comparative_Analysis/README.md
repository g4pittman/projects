# FoodCorp Comparative Analysis — Retail Data & Strategy Project

## 🎯 Objective
Conduct a **comparative performance analysis** of FoodCorp’s four UK retail stores to determine which location should receive a **targeted marketing campaign**.  
This project integrates SQL, Tableau, and Python to measure growth potential, customer loyalty, and store performance using multiple **data-driven KPIs**.

---

## ⚙️ Methodology

### **1. KPI Development**
The analysis centered on six key performance indicators (KPIs), each defined and calculated through SQL queries:

| KPI | Description | Insight |
|------|--------------|----------|
| **Monthly Revenue Over Time** | `SUM(value)` by month and store | Tracks sales seasonality and growth |
| **Monthly Customer Count** | `COUNT(DISTINCT customer_id)` | Measures customer traffic per store |
| **Retention Rate** | Repeat customers ÷ total customers | Evaluates customer loyalty |
| **Churn Rate** | Single-visit customers ÷ total customers | Identifies disengagement patterns |
| **Customer Value Segmentation** | High-value (≥£50) vs low-value (<£50) | Differentiates valuable shoppers |
| **Regional Market Penetration** | Unique shoppers ÷ city population | Gauges growth opportunity:contentReference[oaicite:0]{index=0} |

Each KPI was visualized using **Tableau** for trend analysis and storytelling dashboards.

---

### **2. Data Pipeline**
- Source: FoodCorp retail database (`datas50`)  
- Tables joined: `receipts`, `receipt_lines`, `customers`, and `stores`
- Temporal aggregation performed in **SQL**
- Additional city-level data (population, income, spending) integrated from **UK Office for National Statistics**

Example SQL snippet for revenue KPI:
```sql
SELECT  
    DATE_TRUNC('month', r.purchased_at) AS month,
    r.store_code,
    SUM(rl.value) AS revenue
FROM datas50.receipts r
JOIN datas50.receipt_lines rl ON r.receipt_id = rl.receipt_id
GROUP BY 1, 2
ORDER BY month;

---

## 📊 Visuals

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/FoodCorp_Comparative_Analysis/Screenshot%202025-10-11%20at%206.39.52%20PM.png?raw=true" width="650"/>
</p>

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/FoodCorp_Comparative_Analysis/Screenshot%202025-10-11%20at%206.40.05%20PM.png?raw=true" width="650"/>
</p>


