-- Databricks notebook source
-- MAGIC %md
-- MAGIC #Initialization

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import urllib
-- MAGIC reset = True
-- MAGIC data_id = 93
-- MAGIC exec(urllib.request.urlopen('https://drive.google.com/uc?export=download&id=151MlxWk3Nk-Q3emLDyOFRxPA8lCqgGt-').read())

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Creating a Churn Label - 28 Days Since Last Purchase

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW customer_last_purchase AS
SELECT
    customer_id,
    MAX(purchased_at) AS last_purchase_date
FROM receipts
GROUP BY customer_id;


-- COMMAND ----------

--Creating a Clean Table of Customer Purchases, one row per shop per customer, concerned with 1 date per shop

CREATE OR REPLACE TEMP VIEW customer_purchase_history AS
SELECT
    customer_id,
    CAST(purchased_at AS DATE) AS purchase_date
FROM receipts
WHERE customer_id IS NOT NULL
ORDER BY customer_id, purchase_date;


-- COMMAND ----------

--Looking at Min and max purchase dates (the beginnning and end of the dataset)

SELECT 
    MIN(CAST(purchased_at AS DATE)) AS min_date,
    MAX(CAST(purchased_at AS DATE)) AS max_date
FROM receipts;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC Our data spans from 22nd of July 2020 to 22nd of March 2022

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Window Justification

-- COMMAND ----------

-- MAGIC %md
-- MAGIC In order to balance the need for predictive generalization with data recency, we chose to implement a 14-week tumbling window strategy. Using all available windows (~75) would dilute model performance due to outdated customer behaviors, while using too few would risk underfitting and unstable evaluation. Our window selection starts from July 2021, well within the stable operating period of the dataset, and allows us to allocate 8 windows for training, 3 for validation, and 3 for test — enabling robust tuning and evaluation on temporally unseen data.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC Tumbling Windows:
-- MAGIC
-- MAGIC 14 discrete prediction_dates, spaced 7 days apart
-- MAGIC
-- MAGIC No overlap between windows
-- MAGIC
-- MAGIC Each window is a self-contained prediction snapshot
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # We are creating our windows here. The first tarts at July 1, 2021, Adds 13 more dates, one every 7 days
-- MAGIC
-- MAGIC #Creates a table called prediction_dates for future joins
-- MAGIC
-- MAGIC from pyspark.sql.functions import explode, sequence, to_date, lit
-- MAGIC from pyspark.sql.types import DateType
-- MAGIC from datetime import datetime
-- MAGIC
-- MAGIC # Parameters
-- MAGIC start_date = "2021-07-01"
-- MAGIC num_weeks = 14
-- MAGIC interval_days = 7
-- MAGIC
-- MAGIC # Create DataFrame with 14 weekly prediction dates
-- MAGIC weekly_dates = spark.sql(f"""
-- MAGIC     SELECT explode(
-- MAGIC         sequence(
-- MAGIC             to_date('{start_date}'), 
-- MAGIC             to_date('{start_date}') + INTERVAL {interval_days * (num_weeks - 1)} DAYS, 
-- MAGIC             INTERVAL {interval_days} DAYS
-- MAGIC         )
-- MAGIC     ) AS prediction_date
-- MAGIC """)
-- MAGIC
-- MAGIC weekly_dates.createOrReplaceTempView("prediction_dates")
-- MAGIC
-- MAGIC # Show output
-- MAGIC weekly_dates.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Feature Engineering (Using SQL)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## visits_30days
-- MAGIC The number of receipts (i.e. shopping visits) a customer made in the 30 days before each prediction date.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW visits_30days AS
SELECT
    pd.prediction_date,
    r.customer_id,
    COUNT(DISTINCT r.receipt_id) AS visits_30days
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##total_spend_30days
-- MAGIC
-- MAGIC Total value spent by the customer across all purchases in the 30 days before the prediction date.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW total_spend_30days AS
SELECT
    pd.prediction_date,
    r.customer_id,
    ROUND(SUM(rl.value), 2) AS total_spend_30days
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
JOIN receipt_lines rl
  ON r.receipt_id = rl.receipt_id
GROUP BY pd.prediction_date, r.customer_id;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##days_since_last_visit
-- MAGIC For each (customer_id, prediction_date) pair, calculate how many days have passed since the customer's most recent purchase before the prediction date.
-- MAGIC
-- MAGIC - This tells us how "cold" a customer is at that moment — great signal for churn risk.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW days_since_last_visit AS
SELECT
    pd.prediction_date,
    r.customer_id,
    MIN(DATEDIFF(pd.prediction_date, r.purchased_at)) AS days_since_last_visit
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at < pd.prediction_date
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##avg_basket_value
-- MAGIC
-- MAGIC Average spend per visit in the 30-day window before each prediction date.
-- MAGIC
-- MAGIC - total spend in previous 30 days / # of receipts in previous 30 days

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW avg_basket_value AS
SELECT
    pd.prediction_date,
    r.customer_id,
    ROUND(SUM(rl.value) / COUNT(DISTINCT r.receipt_id), 2) AS avg_basket_value
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
JOIN receipt_lines rl
  ON rl.receipt_id = r.receipt_id
GROUP BY pd.prediction_date, r.customer_id
HAVING COUNT(DISTINCT r.receipt_id) > 0;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Median days between visits
-- MAGIC
-- MAGIC Calculate, for each customer and prediction date, the median number of days between visits over the 30-day window before the prediction.

-- COMMAND ----------

-- Step 1: get visit dates and lag
CREATE OR REPLACE TEMP VIEW visit_lags AS
SELECT
    pd.prediction_date,
    r.customer_id,
    CAST(r.purchased_at AS DATE) AS visit_date,
    LAG(CAST(r.purchased_at AS DATE)) OVER (
        PARTITION BY pd.prediction_date, r.customer_id
        ORDER BY r.purchased_at
    ) AS prev_visit_date
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1);

-- Step 2: compute visit gaps
CREATE OR REPLACE TEMP VIEW visit_gaps AS
SELECT
    prediction_date,
    customer_id,
    DATEDIFF(visit_date, prev_visit_date) AS days_between_visits
FROM visit_lags
WHERE prev_visit_date IS NOT NULL;

-- Step 3: median of those gaps per customer/date
CREATE OR REPLACE TEMP VIEW median_days_between_visits AS
SELECT
    prediction_date,
    customer_id,
    PERCENTILE_APPROX(days_between_visits, 0.5) AS median_days_between_visits
FROM visit_gaps
GROUP BY prediction_date, customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##max_days_between_visits
-- MAGIC
-- MAGIC For each (customer_id, prediction_date), calculate the maximum number of days between visits in the 30-day window prior to the prediction.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW max_days_between_visits AS
SELECT
    prediction_date,
    customer_id,
    MAX(days_between_visits) AS max_days_between_visits
FROM visit_gaps
GROUP BY prediction_date, customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##most_common_department
-- MAGIC
-- MAGIC The department which a shopper shops from the most

-- COMMAND ----------

-- Step 1: Join to get department per line item
CREATE OR REPLACE TEMP VIEW customer_department_counts AS
SELECT
    pd.prediction_date,
    r.customer_id,
    p.department_name,
    SUM(rl.qty) AS total_items
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
JOIN receipt_lines rl
  ON r.receipt_id = rl.receipt_id
JOIN products p
  ON rl.product_code = p.product_code
WHERE p.department_name IS NOT NULL
GROUP BY pd.prediction_date, r.customer_id, p.department_name;

-- Step 2: Rank departments by quantity per customer-date
CREATE OR REPLACE TEMP VIEW department_ranked AS
SELECT *,
       ROW_NUMBER() OVER (
         PARTITION BY prediction_date, customer_id
         ORDER BY total_items DESC
       ) AS dept_rank
FROM customer_department_counts;

-- Step 3: Pick the top-ranked department
CREATE OR REPLACE TEMP VIEW most_common_department AS
SELECT 
    prediction_date,
    customer_id,
    department_name AS most_common_department
FROM department_ranked
WHERE dept_rank = 1;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Avg_items_per_visit
-- MAGIC
-- MAGIC The average number of items the customer purchases each visit

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW avg_items_per_visit AS
SELECT
    pd.prediction_date,
    r.customer_id,
    ROUND(SUM(rl.qty) * 1.0 / COUNT(DISTINCT r.receipt_id), 2) AS avg_items_per_visit
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
JOIN receipt_lines rl
  ON r.receipt_id = rl.receipt_id
GROUP BY pd.prediction_date, r.customer_id
HAVING COUNT(DISTINCT r.receipt_id) > 0;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Unique_categories_30days
-- MAGIC
-- MAGIC The number of unique categories the customer has shopped in in the last 30 days

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW unique_categories_30days AS
SELECT
    pd.prediction_date,
    r.customer_id,
    COUNT(DISTINCT p.category_code) AS unique_categories_30days
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
JOIN receipt_lines rl
  ON r.receipt_id = rl.receipt_id
JOIN products p
  ON rl.product_code = p.product_code
WHERE p.category_code IS NOT NULL
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Age
-- MAGIC
-- MAGIC Age of the customer

-- COMMAND ----------

--displaying age so we can see how it is originally formatted before we change formatting to create 'age' feature

SELECT dob
FROM customers
WHERE dob IS NOT NULL
LIMIT 5;


-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW customer_age AS
SELECT
    pd.prediction_date,
    c.customer_id,
    FLOOR(DATEDIFF(pd.prediction_date, c.dob) / 365.25) AS customer_age
FROM prediction_dates pd
JOIN customers c
  ON c.customer_id IS NOT NULL
WHERE c.dob IS NOT NULL;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##shop_weekend_pct
-- MAGIC
-- MAGIC % of time the customer shops on the weekend
-- MAGIC
-- MAGIC
-- MAGIC Output explained:
-- MAGIC - 0 is a weekday shopper
-- MAGIC - 100 is a weekend only shopper

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW shop_weekend_pct AS
SELECT
    pd.prediction_date,
    r.customer_id,
    ROUND(
        100.0 * SUM(CASE 
                      WHEN DAYOFWEEK(CAST(r.purchased_at AS DATE)) IN (1, 7) THEN 1 
                      ELSE 0 
                   END) / COUNT(DISTINCT r.receipt_id), 
        2
    ) AS shop_weekend_pct
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
GROUP BY pd.prediction_date, r.customer_id
HAVING COUNT(DISTINCT r.receipt_id) > 0;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##habitual_then_stopped
-- MAGIC
-- MAGIC For this one, we are looking at 60 days prior. In the 60-30 day period, did they shop at least 3 times? If they did, and they shopped 0 times in the previous 30 days, then they will recieve a 1. If anything else, 0.
-- MAGIC
-- MAGIC - 1 means they were loyal, but stopped last month

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW habitual_then_stopped AS
SELECT
    pd.prediction_date,
    r.customer_id,
    CASE 
        WHEN COUNT(DISTINCT CASE 
                WHEN r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 60) AND DATE_SUB(pd.prediction_date, 31)
                     THEN r.receipt_id END) >= 3
         AND COUNT(DISTINCT CASE 
                WHEN r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 30) AND DATE_SUB(pd.prediction_date, 1)
                     THEN r.receipt_id END) = 0
        THEN 1 ELSE 0
    END AS habitual_then_stopped
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at BETWEEN DATE_SUB(pd.prediction_date, 60) AND DATE_SUB(pd.prediction_date, 1)
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Creating the first Lag Window (30-60 days)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC lagging variables:
-- MAGIC visits
-- MAGIC
-- MAGIC total spend
-- MAGIC
-- MAGIC unique categories
-- MAGIC
-- MAGIC average basket value 

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW lag1_features AS
SELECT
    pd.prediction_date,
    r.customer_id,
    COUNT(DISTINCT r.receipt_id) AS visits_lag1_30days,
    SUM(rl.value) AS total_spend_lag1_30days,
    COUNT(DISTINCT p.category_code) AS unique_categories_lag1_30days,
    CASE 
        WHEN COUNT(DISTINCT r.receipt_id) > 0 THEN SUM(rl.value) / COUNT(DISTINCT r.receipt_id)
        ELSE 0
    END AS avg_basket_value_lag1_30days
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at > DATE_SUB(pd.prediction_date, 60)
 AND r.purchased_at <= DATE_SUB(pd.prediction_date, 30)
JOIN receipt_lines rl ON r.receipt_id = rl.receipt_id
JOIN products p ON rl.product_code = p.product_code
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Creating the second Lag Window (60-90 days)

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW lag2_features AS
SELECT
    pd.prediction_date,
    r.customer_id,
    COUNT(DISTINCT r.receipt_id) AS visits_lag2_30days,
    SUM(rl.value) AS total_spend_lag2_30days,
    COUNT(DISTINCT p.category_code) AS unique_categories_lag2_30days,
    CASE 
        WHEN COUNT(DISTINCT r.receipt_id) > 0 THEN SUM(rl.value) / COUNT(DISTINCT r.receipt_id)
        ELSE 0
    END AS avg_basket_value_lag2_30days
FROM prediction_dates pd
JOIN receipts r
  ON r.purchased_at > DATE_SUB(pd.prediction_date, 90)
 AND r.purchased_at <= DATE_SUB(pd.prediction_date, 60)
JOIN receipt_lines rl ON r.receipt_id = rl.receipt_id
JOIN products p ON rl.product_code = p.product_code
GROUP BY pd.prediction_date, r.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Churn Label for Each Customer
-- MAGIC For each (customer_id, prediction_date) pair, we want to assign a label:
-- MAGIC
-- MAGIC
-- MAGIC 1 = churned
-- MAGIC 0 = not churned
-- MAGIC

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW churn_labels AS
SELECT
    a.prediction_date,
    a.customer_id,
    CASE 
        WHEN MIN(r.purchased_at) IS NULL THEN 1
        ELSE 0
    END AS churned
FROM (
    -- Step 1: Get all customers who are active at each prediction_date
    SELECT DISTINCT
        r.customer_id,
        pd.prediction_date
    FROM prediction_dates pd
    JOIN receipts r
      ON r.purchased_at < pd.prediction_date
) a

-- Step 2: Left join to find future purchases within 28-day lookahead
LEFT JOIN receipts r
  ON a.customer_id = r.customer_id
 AND r.purchased_at > a.prediction_date
 AND r.purchased_at <= DATE_ADD(a.prediction_date, 28)

GROUP BY a.prediction_date, a.customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Joining the features into one table, with the churn label

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW featured_table AS
SELECT 
    v.prediction_date,
    v.customer_id,
    v.visits_30days,
    s.total_spend_30days,
    d.days_since_last_visit,
    b.avg_basket_value,
    med.median_days_between_visits,
    mx.max_days_between_visits,
    dep.most_common_department,
    aiv.avg_items_per_visit,
    cat.unique_categories_30days,
    age.customer_age,
    wk.shop_weekend_pct,
    hs.habitual_then_stopped

FROM visits_30days v
LEFT JOIN total_spend_30days s
  ON v.prediction_date = s.prediction_date AND v.customer_id = s.customer_id
LEFT JOIN days_since_last_visit d
  ON v.prediction_date = d.prediction_date AND v.customer_id = d.customer_id
LEFT JOIN avg_basket_value b
  ON v.prediction_date = b.prediction_date AND v.customer_id = b.customer_id
LEFT JOIN median_days_between_visits med
  ON v.prediction_date = med.prediction_date AND v.customer_id = med.customer_id
LEFT JOIN max_days_between_visits mx
  ON v.prediction_date = mx.prediction_date AND v.customer_id = mx.customer_id
LEFT JOIN most_common_department dep
  ON v.prediction_date = dep.prediction_date AND v.customer_id = dep.customer_id
LEFT JOIN avg_items_per_visit aiv
  ON v.prediction_date = aiv.prediction_date AND v.customer_id = aiv.customer_id
LEFT JOIN unique_categories_30days cat
  ON v.prediction_date = cat.prediction_date AND v.customer_id = cat.customer_id
LEFT JOIN customer_age age
  ON v.prediction_date = age.prediction_date AND v.customer_id = age.customer_id
LEFT JOIN shop_weekend_pct wk
  ON v.prediction_date = wk.prediction_date AND v.customer_id = wk.customer_id
LEFT JOIN habitual_then_stopped hs
  ON v.prediction_date = hs.prediction_date AND v.customer_id = hs.customer_id;



-- COMMAND ----------

--Viewing the joined table with all joined featurese
SELECT * FROM featured_table ORDER BY prediction_date DESC LIMIT 10;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Attaching the churn label to the table

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW prepared_data AS
SELECT
    f.*,
    l.churned
FROM featured_table f
LEFT JOIN churn_labels l
  ON f.prediction_date = l.prediction_date
 AND f.customer_id = l.customer_id;



-- COMMAND ----------

--looking at total churn count, to make sure we have done it right and have a % of churn to non-churn that makes sense

SELECT churned, COUNT(*) 
FROM prepared_data 
GROUP BY churned;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Joinging the lagged features

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW enriched_featured_table AS
SELECT p.*,
       l1.visits_lag1_30days,
       l1.total_spend_lag1_30days,
       l1.unique_categories_lag1_30days,
       l1.avg_basket_value_lag1_30days,
       l2.visits_lag2_30days,
       l2.total_spend_lag2_30days,
       l2.unique_categories_lag2_30days,
       l2.avg_basket_value_lag2_30days
FROM prepared_data p
LEFT JOIN lag1_features l1
  ON p.prediction_date = l1.prediction_date
 AND p.customer_id = l1.customer_id
LEFT JOIN lag2_features l2
  ON p.prediction_date = l2.prediction_date
 AND p.customer_id = l2.customer_id;


-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW final_model_data AS
SELECT *
FROM enriched_featured_table;


-- COMMAND ----------

SELECT COUNT(*) AS total_rows,
       SUM(CASE WHEN visits_lag1_30days IS NULL THEN 1 ELSE 0 END) AS null_lag1,
       SUM(CASE WHEN churned IS NULL THEN 1 ELSE 0 END) AS null_labels
FROM final_model_data;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Creating Delta Variables

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW final_model_data_with_deltas AS
SELECT *,
    -- Delta Features (0–30 minus 30–60)
    visits_30days - visits_lag1_30days AS delta_visits,
    total_spend_30days - total_spend_lag1_30days AS delta_total_spend,
    unique_categories_30days - unique_categories_lag1_30days AS delta_unique_categories,
    avg_basket_value - avg_basket_value_lag1_30days AS delta_avg_basket_value
FROM final_model_data;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Train, Validation, and Test Split

-- COMMAND ----------

SELECT MIN(prediction_date), MAX(prediction_date) FROM final_model_data;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Adding Labels (Train, Validation, or Split)

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW split_data AS
SELECT *,
    CASE 
        WHEN prediction_date < DATE('2021-08-20') THEN 'train'
        WHEN prediction_date < DATE('2021-09-10') THEN 'val'
        ELSE 'test'
    END AS data_split
FROM final_model_data_with_deltas;



-- COMMAND ----------

--checking count of each group, to ensure we have ample data in each and that the labels sorted themselved out correctly
SELECT data_split, COUNT(*) FROM split_data GROUP BY data_split;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Validating Train, Validation, and Test Data

-- COMMAND ----------

SELECT *
FROM split_data
ORDER BY prediction_date DESC
LIMIT 10;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Training Baseline Model
-- MAGIC - using only 1 feature to create a baseline model
-- MAGIC - creating evaluation metrics, so that we can compare future models and ensure that they are building from this simple baseline

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Model: Logistic Regression
-- MAGIC
-- MAGIC Feature: days_since_last_visit
-- MAGIC
-- MAGIC Evaluation Metrics: Accruacy, Precision, Recall, F1-Score, AUC-ROC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Metric --	Why It's Useful
-- MAGIC
-- MAGIC
-- MAGIC Accuracy	Simple, intuitive, but can mislead if data is imbalanced
-- MAGIC
-- MAGIC
-- MAGIC Precision	How reliable your churn predictions are
-- MAGIC
-- MAGIC
-- MAGIC Recall	How well you catch all churners
-- MAGIC
-- MAGIC
-- MAGIC F1 Score	Balance between precision and recall
-- MAGIC
-- MAGIC
-- MAGIC AUC-ROC	Measures how well the model ranks churners vs non-churners

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC # Load data from Spark into pandas
-- MAGIC df = spark.table("split_data").toPandas()
-- MAGIC
-- MAGIC # Drop rows with missing values just in case
-- MAGIC df = df.dropna(subset=["days_since_last_visit", "churned", "data_split"])
-- MAGIC
-- MAGIC # Split data
-- MAGIC train_df = df[df["data_split"] == "train"]
-- MAGIC val_df   = df[df["data_split"] == "val"]
-- MAGIC test_df  = df[df["data_split"] == "test"]
-- MAGIC
-- MAGIC # Feature and target
-- MAGIC X_train = train_df[["days_since_last_visit"]]
-- MAGIC y_train = train_df["churned"]
-- MAGIC
-- MAGIC X_val = val_df[["days_since_last_visit"]]
-- MAGIC y_val = val_df["churned"]
-- MAGIC
-- MAGIC X_test = test_df[["days_since_last_visit"]]
-- MAGIC y_test = test_df["churned"]
-- MAGIC
-- MAGIC # Fit logistic regression
-- MAGIC from sklearn.linear_model import LogisticRegression
-- MAGIC model = LogisticRegression()
-- MAGIC model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Predict on test set
-- MAGIC y_pred = model.predict(X_val)
-- MAGIC y_proba = model.predict_proba(X_val)[:, 1]  # Probabilities for ROC
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
-- MAGIC
-- MAGIC # Basic classification metrics
-- MAGIC print("Baseline Logistic Regression (1 Feature)")
-- MAGIC print("-----------------------------------------")
-- MAGIC print(f"Accuracy:  {accuracy_score(y_val, y_pred):.3f}")
-- MAGIC print(f"Precision: {precision_score(y_val, y_pred):.3f}")
-- MAGIC print(f"Recall:    {recall_score(y_val, y_pred):.3f}")
-- MAGIC print(f"F1 Score:  {f1_score(y_val, y_pred):.3f}")
-- MAGIC print(f"AUC-ROC:   {roc_auc_score(y_val, y_proba):.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Feature Selection -- Permutation

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC df = spark.table("split_data").toPandas()
-- MAGIC
-- MAGIC # Impute all numeric features to avoid dropping rows
-- MAGIC default_imputations = {
-- MAGIC     "avg_items_per_visit": 0,
-- MAGIC     "total_spend_30days": 0,
-- MAGIC     "shop_weekend_pct": 0,
-- MAGIC     "unique_categories_30days": 0,
-- MAGIC     "habitual_then_stopped": 0,
-- MAGIC     "median_days_between_visits": 30,
-- MAGIC     "max_days_between_visits": 30,
-- MAGIC     "visits_lag1_30days": 0,
-- MAGIC     "visits_lag2_30days": 0,
-- MAGIC     "total_spend_lag1_30days": 0,
-- MAGIC     "total_spend_lag2_30days": 0,
-- MAGIC     "unique_categories_lag1_30days": 0,
-- MAGIC     "unique_categories_lag2_30days": 0,
-- MAGIC     "avg_basket_value_lag1_30days": 0,
-- MAGIC     "avg_basket_value_lag2_30days": 0,
-- MAGIC     "most_common_department": "Unknown"
-- MAGIC }
-- MAGIC
-- MAGIC df = df.fillna(default_imputations)
-- MAGIC
-- MAGIC # One-hot encode department
-- MAGIC df = pd.get_dummies(df, columns=["most_common_department"], drop_first=True)
-- MAGIC
-- MAGIC # NOW define features (after encoding)
-- MAGIC feature_cols = [col for col in df.columns if col not in ["prediction_date", "customer_id", "churned", "data_split"]]
-- MAGIC
-- MAGIC # Final pass: remove remaining infs or NaNs
-- MAGIC import numpy as np
-- MAGIC df = df.replace([np.inf, -np.inf], np.nan)
-- MAGIC df[feature_cols] = df[feature_cols].fillna(0)  # final safety net
-- MAGIC
-- MAGIC # Scale features
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC scaler = StandardScaler()
-- MAGIC df[feature_cols] = scaler.fit_transform(df[feature_cols])
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC print("Features used in initial model:")
-- MAGIC for col in feature_cols:
-- MAGIC     print(col)
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC # Train/test split
-- MAGIC X_train = df[df["data_split"] == "train"][feature_cols]
-- MAGIC y_train = df[df["data_split"] == "train"]["churned"]
-- MAGIC X_val = df[df["data_split"] == "val"][feature_cols]
-- MAGIC y_val = df[df["data_split"] == "val"]["churned"]
-- MAGIC
-- MAGIC # Train model
-- MAGIC from sklearn.linear_model import LogisticRegression
-- MAGIC model = LogisticRegression(max_iter=1000)
-- MAGIC model.fit(X_train, y_train)
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC from sklearn.inspection import permutation_importance
-- MAGIC
-- MAGIC # Refit model with updated features
-- MAGIC X_train = df[df["data_split"] == "train"][feature_cols]
-- MAGIC y_train = df[df["data_split"] == "train"]["churned"]
-- MAGIC X_val = df[df["data_split"] == "val"][feature_cols]
-- MAGIC y_val = df[df["data_split"] == "val"]["churned"]
-- MAGIC
-- MAGIC model = LogisticRegression(max_iter=1000)
-- MAGIC model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Permutation importance on validation set
-- MAGIC result = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)
-- MAGIC
-- MAGIC # Plot
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import numpy as np
-- MAGIC
-- MAGIC importances = result.importances_mean
-- MAGIC indices = np.argsort(importances)[::-1]
-- MAGIC top_n = 40  # top 40 most important
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 8))
-- MAGIC plt.barh(np.array(feature_cols)[indices[:top_n]], importances[indices[:top_n]])
-- MAGIC plt.xlabel("Mean Importance (drop in AUC)")
-- MAGIC plt.title("Permutation Feature Importance (Validation ROC-AUC)")
-- MAGIC plt.gca().invert_yaxis()
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Pre-permutation deletion Score

-- COMMAND ----------

-- MAGIC %py
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Ensure you've already predicted these:
-- MAGIC # y_test: true churn values for test set
-- MAGIC # y_pred: predicted class labels
-- MAGIC # y_proba: predicted probabilities for class 1 (churned)
-- MAGIC
-- MAGIC # Calculate metrics
-- MAGIC balanced_acc = balanced_accuracy_score(y_val, y_pred)
-- MAGIC roc_auc = roc_auc_score(y_val, y_proba)
-- MAGIC
-- MAGIC # Display results
-- MAGIC print("Model Evaluation Metrics (Val Set)")
-- MAGIC print("=====================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Getting rid of features that permutation score showed as unimportant, rerunning and seeing how our score changes

-- COMMAND ----------

-- MAGIC
-- MAGIC %py
-- MAGIC
-- MAGIC #running the same regression, but this time deleting the features that permutation informed us were unimportant
-- MAGIC
-- MAGIC from sklearn.linear_model import LogisticRegression
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC import pandas as pd
-- MAGIC import numpy as np
-- MAGIC
-- MAGIC # Load the table from Spark
-- MAGIC df = spark.table("split_data").toPandas()
-- MAGIC
-- MAGIC # Impute missing values
-- MAGIC default_imputations = {
-- MAGIC     "avg_items_per_visit": 0,
-- MAGIC     "total_spend_30days": 0,
-- MAGIC     "shop_weekend_pct": 0,
-- MAGIC     "unique_categories_30days": 0,
-- MAGIC     "habitual_then_stopped": 0,
-- MAGIC     "median_days_between_visits": 30,
-- MAGIC     "max_days_between_visits": 30,
-- MAGIC     "visits_lag1_30days": 0,
-- MAGIC     "visits_lag2_30days": 0,
-- MAGIC     "total_spend_lag1_30days": 0,
-- MAGIC     "total_spend_lag2_30days": 0,
-- MAGIC     "unique_categories_lag1_30days": 0,
-- MAGIC     "unique_categories_lag2_30days": 0,
-- MAGIC     "avg_basket_value_lag1_30days": 0,
-- MAGIC     "avg_basket_value_lag2_30days": 0,
-- MAGIC     "delta_visits": 0,
-- MAGIC     "delta_total_spend": 0,
-- MAGIC     "delta_avg_basket_value": 0,
-- MAGIC     "delta_unique_categories": 0,
-- MAGIC     "customer_age": df["customer_age"].median(),
-- MAGIC     "most_common_department": "Unknown"
-- MAGIC }
-- MAGIC df = df.fillna(default_imputations)
-- MAGIC
-- MAGIC # One-hot encode the department feature
-- MAGIC df = pd.get_dummies(df, columns=["most_common_department"], drop_first=True)
-- MAGIC
-- MAGIC # Drop unwanted features
-- MAGIC drop_cols = ["customer_age", "habitual_then_stopped"]
-- MAGIC drop_cols += [col for col in df.columns if col.startswith("most_common_department_")]
-- MAGIC df = df.drop(columns=drop_cols)
-- MAGIC
-- MAGIC # Define feature columns
-- MAGIC feature_cols = [col for col in df.columns if col not in ["prediction_date", "customer_id", "churned", "data_split"]]
-- MAGIC
-- MAGIC # Final cleanup
-- MAGIC df = df.replace([np.inf, -np.inf], np.nan)
-- MAGIC df[feature_cols] = df[feature_cols].fillna(0)
-- MAGIC
-- MAGIC # Scale the features
-- MAGIC scaler = StandardScaler()
-- MAGIC df[feature_cols] = scaler.fit_transform(df[feature_cols])
-- MAGIC
-- MAGIC # Train-test-validation split
-- MAGIC X_train = df[df["data_split"] == "train"][feature_cols]
-- MAGIC y_train = df[df["data_split"] == "train"]["churned"]
-- MAGIC
-- MAGIC X_val = df[df["data_split"] == "val"][feature_cols]
-- MAGIC y_val = df[df["data_split"] == "val"]["churned"]
-- MAGIC
-- MAGIC X_test = df[df["data_split"] == "test"][feature_cols]
-- MAGIC y_test = df[df["data_split"] == "test"]["churned"]
-- MAGIC
-- MAGIC # Train the logistic regression model
-- MAGIC model = LogisticRegression(max_iter=1000)
-- MAGIC model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Predict on validation set
-- MAGIC y_pred_val = model.predict(X_val)
-- MAGIC y_proba_val = model.predict_proba(X_val)[:, 1]
-- MAGIC
-- MAGIC # Evaluate
-- MAGIC balanced_acc_val = balanced_accuracy_score(y_val, y_pred_val)
-- MAGIC roc_auc_val = roc_auc_score(y_val, y_proba_val)
-- MAGIC
-- MAGIC print("Logistic Regression Performance (Validation Set)")
-- MAGIC print("=================================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_val:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_val:.3f}")
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC
-- MAGIC print("Row counts used in the model:")
-- MAGIC print(f"Train: {len(X_train)}")
-- MAGIC print(f"Val:   {len(X_val)}")
-- MAGIC print(f"Test:  {len(X_test)}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Model Testing Using Random Forest

-- COMMAND ----------

-- MAGIC %py
-- MAGIC from sklearn.ensemble import RandomForestClassifier
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Train the model on training set
-- MAGIC rf_model = RandomForestClassifier(
-- MAGIC     n_estimators=100,
-- MAGIC     max_depth=None,
-- MAGIC     random_state=42,
-- MAGIC     class_weight='balanced'
-- MAGIC )
-- MAGIC rf_model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Predict on validation set (NOT test set)
-- MAGIC y_pred_rf_val = rf_model.predict(X_val)
-- MAGIC y_proba_rf_val = rf_model.predict_proba(X_val)[:, 1]
-- MAGIC
-- MAGIC # Evaluate on validation set
-- MAGIC balanced_acc_rf_val = balanced_accuracy_score(y_val, y_pred_rf_val)
-- MAGIC roc_auc_rf_val = roc_auc_score(y_val, y_proba_rf_val)
-- MAGIC
-- MAGIC # Display scores
-- MAGIC print("Random Forest Performance (Validation Set)")
-- MAGIC print("===========================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_rf_val:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_rf_val:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Training Data on XGBoost

-- COMMAND ----------

-- MAGIC %sh
-- MAGIC pip install xgboost

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import xgboost as xgb
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Initialize the classifier
-- MAGIC xgb_model = xgb.XGBClassifier(
-- MAGIC     n_estimators=100,
-- MAGIC     max_depth=6,
-- MAGIC     learning_rate=0.1,
-- MAGIC     subsample=0.8,
-- MAGIC     colsample_bytree=0.8,
-- MAGIC     use_label_encoder=False,
-- MAGIC     eval_metric='logloss',
-- MAGIC     random_state=42
-- MAGIC )
-- MAGIC
-- MAGIC # Train on training set
-- MAGIC xgb_model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Predict on validation set
-- MAGIC y_pred_xgb_val = xgb_model.predict(X_val)
-- MAGIC y_proba_xgb_val = xgb_model.predict_proba(X_val)[:, 1]
-- MAGIC
-- MAGIC # Evaluate
-- MAGIC balanced_acc_xgb_val = balanced_accuracy_score(y_val, y_pred_xgb_val)
-- MAGIC roc_auc_xgb_val = roc_auc_score(y_val, y_proba_xgb_val)
-- MAGIC
-- MAGIC # Display results
-- MAGIC print("XGBoost Performance (Validation Set)")
-- MAGIC print("====================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_xgb_val:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_xgb_val:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Hyper-Parameter Tuning for XGBoost
-- MAGIC
-- MAGIC -testing different combinations of hyperparameters to see which gets the best ROCAUC and Balanced Accuracy Score

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import xgboost as xgb
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC import itertools
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC # Define the grid
-- MAGIC param_grid = {
-- MAGIC     "max_depth": [3, 5, 7, 9],
-- MAGIC     "learning_rate": [0.01, 0.05, 0.1],
-- MAGIC     "n_estimators": [100, 200, 300],
-- MAGIC     "subsample": [0.8],
-- MAGIC     "colsample_bytree": [0.8]
-- MAGIC }
-- MAGIC
-- MAGIC # Track results
-- MAGIC results = []
-- MAGIC
-- MAGIC # Manual grid search
-- MAGIC for max_depth, learning_rate, n_estimators in itertools.product(
-- MAGIC     param_grid["max_depth"],
-- MAGIC     param_grid["learning_rate"],
-- MAGIC     param_grid["n_estimators"]
-- MAGIC ):
-- MAGIC     model = xgb.XGBClassifier(
-- MAGIC         max_depth=max_depth,
-- MAGIC         learning_rate=learning_rate,
-- MAGIC         n_estimators=n_estimators,
-- MAGIC         subsample=0.8,
-- MAGIC         colsample_bytree=0.8,
-- MAGIC         use_label_encoder=False,
-- MAGIC         eval_metric="logloss",
-- MAGIC         random_state=42
-- MAGIC     )
-- MAGIC
-- MAGIC     model.fit(X_train, y_train)
-- MAGIC     y_pred_val = model.predict(X_val)
-- MAGIC     y_proba_val = model.predict_proba(X_val)[:, 1]
-- MAGIC
-- MAGIC     bal_acc = balanced_accuracy_score(y_val, y_pred_val)
-- MAGIC     auc = roc_auc_score(y_val, y_proba_val)
-- MAGIC
-- MAGIC     results.append({
-- MAGIC         "max_depth": max_depth,
-- MAGIC         "learning_rate": learning_rate,
-- MAGIC         "n_estimators": n_estimators,
-- MAGIC         "balanced_accuracy": bal_acc,
-- MAGIC         "roc_auc": auc,
-- MAGIC         "model": model
-- MAGIC     })
-- MAGIC
-- MAGIC # Convert results to DataFrame and sort by ROC-AUC
-- MAGIC results_df = pd.DataFrame(results)
-- MAGIC results_df_sorted = results_df.sort_values(by="roc_auc", ascending=False)
-- MAGIC
-- MAGIC # Show top 10 models by ROC-AUC
-- MAGIC print("Top 10 XGBoost Models by ROC-AUC (Validation Set):")
-- MAGIC print(results_df_sorted[["max_depth", "learning_rate", "n_estimators", "balanced_accuracy", "roc_auc"]].head(10))
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Optimized XGBoost with tuned Hyperparameters

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC df = spark.table("split_data").toPandas()
-- MAGIC
-- MAGIC # Impute missing values
-- MAGIC default_imputations = {
-- MAGIC     "avg_items_per_visit": 0,
-- MAGIC     "total_spend_30days": 0,
-- MAGIC     "shop_weekend_pct": 0,
-- MAGIC     "unique_categories_30days": 0,
-- MAGIC     "habitual_then_stopped": 0,
-- MAGIC     "median_days_between_visits": 30,
-- MAGIC     "max_days_between_visits": 30,
-- MAGIC     "visits_lag1_30days": 0,
-- MAGIC     "visits_lag2_30days": 0,
-- MAGIC     "total_spend_lag1_30days": 0,
-- MAGIC     "total_spend_lag2_30days": 0,
-- MAGIC     "unique_categories_lag1_30days": 0,
-- MAGIC     "unique_categories_lag2_30days": 0,
-- MAGIC     "avg_basket_value_lag1_30days": 0,
-- MAGIC     "avg_basket_value_lag2_30days": 0,
-- MAGIC     "delta_visits": 0,
-- MAGIC     "delta_total_spend": 0,
-- MAGIC     "delta_avg_basket_value": 0,
-- MAGIC     "delta_unique_categories": 0,
-- MAGIC     "customer_age": df["customer_age"].median(),
-- MAGIC     "most_common_department": "Unknown"
-- MAGIC }
-- MAGIC df = df.fillna(default_imputations)
-- MAGIC
-- MAGIC # One-hot encode department
-- MAGIC df = pd.get_dummies(df, columns=["most_common_department"], drop_first=True)
-- MAGIC
-- MAGIC # Drop irrelevant features
-- MAGIC drop_cols = ["customer_age", "habitual_then_stopped"]
-- MAGIC drop_cols += [col for col in df.columns if col.startswith("most_common_department_")]
-- MAGIC df = df.drop(columns=drop_cols)
-- MAGIC
-- MAGIC # Define features
-- MAGIC feature_cols = [col for col in df.columns if col not in ["prediction_date", "customer_id", "churned", "data_split"]]
-- MAGIC df = df.replace([np.inf, -np.inf], np.nan)
-- MAGIC df[feature_cols] = df[feature_cols].fillna(0)
-- MAGIC
-- MAGIC # Scale
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC scaler = StandardScaler()
-- MAGIC df[feature_cols] = scaler.fit_transform(df[feature_cols])
-- MAGIC
-- MAGIC # Train-test split
-- MAGIC X_train = df[df["data_split"] == "train"][feature_cols]
-- MAGIC y_train = df[df["data_split"] == "train"]["churned"]
-- MAGIC X_test = df[df["data_split"] == "test"][feature_cols]
-- MAGIC y_test = df[df["data_split"] == "test"]["churned"]
-- MAGIC
-- MAGIC # Train optimized XGBoost model
-- MAGIC from xgboost import XGBClassifier
-- MAGIC model = XGBClassifier(
-- MAGIC     max_depth=3,
-- MAGIC     reg_alpha=1, #adds L1 regularization penalty
-- MAGIC     sub_sample=0.8, #another regularization step
-- MAGIC     colsample_bytree=0.8, #" "
-- MAGIC     learning_rate=0.05,
-- MAGIC     n_estimators=300,
-- MAGIC     use_label_encoder=False,
-- MAGIC     eval_metric='logloss'
-- MAGIC )
-- MAGIC model.fit(X_train, y_train)
-- MAGIC
-- MAGIC # Evaluate
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC y_pred = model.predict(X_test)
-- MAGIC y_proba = model.predict_proba(X_test)[:, 1]
-- MAGIC balanced_acc = balanced_accuracy_score(y_test, y_pred)
-- MAGIC roc_auc = roc_auc_score(y_test, y_proba)
-- MAGIC
-- MAGIC print("Optimized XGBoost Model (Validation Set)")
-- MAGIC print("="*60)
-- MAGIC print(f"Balanced Accuracy: {balanced_acc:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC This is the score of the final model using the validation (for comparison on test set scores)

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Filter to validation set
-- MAGIC df_val = df[df["data_split"] == "val"].copy()
-- MAGIC X_val = df_val[feature_cols]
-- MAGIC y_val = df_val["churned"]
-- MAGIC
-- MAGIC # Predict
-- MAGIC df_val["churn_probability"] = model.predict_proba(X_val)[:, 1]
-- MAGIC df_val["churn_pred"] = (df_val["churn_probability"] >= 0.5).astype(int)
-- MAGIC
-- MAGIC # Evaluate
-- MAGIC balanced_acc_val = balanced_accuracy_score(y_val, df_val["churn_pred"])
-- MAGIC roc_auc_val = roc_auc_score(y_val, df_val["churn_probability"])
-- MAGIC
-- MAGIC print("XGBoost Model Performance (Validation Set)")
-- MAGIC print("==========================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_val:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_val:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Final Model Predictions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Valudation Score:

-- COMMAND ----------

-- MAGIC %py
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Filter to validation set
-- MAGIC df_val = df[df["data_split"] == "val"].copy()
-- MAGIC X_val = df_val[feature_cols]
-- MAGIC y_val = df_val["churned"]
-- MAGIC
-- MAGIC # Predict with the final XGBoost model (stored as 'model')
-- MAGIC df_val["churn_probability"] = model.predict_proba(X_val)[:, 1]
-- MAGIC df_val["churn_pred"] = (df_val["churn_probability"] >= 0.5).astype(int)
-- MAGIC
-- MAGIC # Evaluate performance
-- MAGIC balanced_acc_val = balanced_accuracy_score(y_val, df_val["churn_pred"])
-- MAGIC roc_auc_val = roc_auc_score(y_val, df_val["churn_probability"])
-- MAGIC
-- MAGIC print("Final XGBoost Model Performance (Validation Set)")
-- MAGIC print("=================================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_val:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_val:.3f}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Test Score:

-- COMMAND ----------

-- MAGIC %py
-- MAGIC from sklearn.metrics import balanced_accuracy_score, roc_auc_score
-- MAGIC
-- MAGIC # Filter to test set
-- MAGIC df_test = df[df["data_split"] == "test"].copy()
-- MAGIC X_test = df_test[feature_cols]
-- MAGIC y_test = df_test["churned"]
-- MAGIC
-- MAGIC # Predict with the final XGBoost model (stored as 'model')
-- MAGIC df_test["churn_probability"] = model.predict_proba(X_test)[:, 1]
-- MAGIC df_test["churn_pred"] = (df_test["churn_probability"] >= 0.5).astype(int)
-- MAGIC
-- MAGIC # Evaluate performance
-- MAGIC balanced_acc_test = balanced_accuracy_score(y_test, df_test["churn_pred"])
-- MAGIC roc_auc_test = roc_auc_score(y_test, df_test["churn_probability"])
-- MAGIC
-- MAGIC print("Final XGBoost Model Performance (Test Set)")
-- MAGIC print("===========================================")
-- MAGIC print(f"Balanced Accuracy: {balanced_acc_test:.3f}")
-- MAGIC print(f"ROC-AUC Score:     {roc_auc_test:.3f}")
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Training w/ Validation vs. Test Error Score

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC metrics = ["Balanced Accuracy", "ROC-AUC"]
-- MAGIC val_scores = [balanced_acc_val, roc_auc_val]
-- MAGIC test_scores = [balanced_acc_test, roc_auc_test]
-- MAGIC
-- MAGIC x = range(len(metrics))
-- MAGIC plt.figure(figsize=(6, 4))
-- MAGIC plt.bar(x, val_scores, width=0.35, label="Validation", align='center')
-- MAGIC plt.bar([i + 0.4 for i in x], test_scores, width=0.35, label="Test", align='center')
-- MAGIC plt.xticks([i + 0.2 for i in x], metrics)
-- MAGIC plt.ylabel("Score")
-- MAGIC plt.title("Validation vs Test Performance")
-- MAGIC plt.legend()
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Finding the initial Learning Curve for XGBoost

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC
-- MAGIC import numpy as np
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC from xgboost import XGBClassifier
-- MAGIC from sklearn.metrics import roc_auc_score
-- MAGIC
-- MAGIC # Prepare your datasets
-- MAGIC df_train = df[df["data_split"] == "train"].copy()
-- MAGIC df_val   = df[df["data_split"] == "val"].copy()
-- MAGIC df_test  = df[df["data_split"] == "test"].copy()
-- MAGIC
-- MAGIC X_train_full = df_train[feature_cols]
-- MAGIC y_train_full = df_train["churned"]
-- MAGIC
-- MAGIC X_val = df_val[feature_cols]
-- MAGIC y_val = df_val["churned"]
-- MAGIC
-- MAGIC X_test = df_test[feature_cols]
-- MAGIC y_test = df_test["churned"]
-- MAGIC
-- MAGIC # Define learning curve sizes (e.g., 10% to 100%)
-- MAGIC train_sizes = np.linspace(0.1, 1.0, 10)
-- MAGIC
-- MAGIC # Store results
-- MAGIC train_scores = []
-- MAGIC val_scores = []
-- MAGIC test_scores = []
-- MAGIC
-- MAGIC for frac in train_sizes:
-- MAGIC     # Subsample training data
-- MAGIC     sample_size = int(frac * len(X_train_full))
-- MAGIC     X_train_sample = X_train_full.iloc[:sample_size]
-- MAGIC     y_train_sample = y_train_full.iloc[:sample_size]
-- MAGIC
-- MAGIC     # Train model
-- MAGIC     model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
-- MAGIC     model.fit(X_train_sample, y_train_sample)
-- MAGIC
-- MAGIC     # Predict probabilities
-- MAGIC     y_train_prob = model.predict_proba(X_train_sample)[:, 1]
-- MAGIC     y_val_prob = model.predict_proba(X_val)[:, 1]
-- MAGIC     y_test_prob = model.predict_proba(X_test)[:, 1]
-- MAGIC
-- MAGIC     # Compute ROC-AUC
-- MAGIC     train_scores.append(roc_auc_score(y_train_sample, y_train_prob))
-- MAGIC     val_scores.append(roc_auc_score(y_val, y_val_prob))
-- MAGIC     test_scores.append(roc_auc_score(y_test, y_test_prob))
-- MAGIC
-- MAGIC # Plotting
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.plot(train_sizes * len(X_train_full), train_scores, 'o-', label="Train ROC-AUC")
-- MAGIC plt.plot(train_sizes * len(X_train_full), val_scores, 'o-', label="Validation ROC-AUC")
-- MAGIC plt.plot(train_sizes * len(X_train_full), test_scores, 'o-', label="Test ROC-AUC")
-- MAGIC
-- MAGIC plt.xlabel("Training Set Size")
-- MAGIC plt.ylabel("ROC-AUC Score")
-- MAGIC plt.title("Learning Curve (XGBoost)")
-- MAGIC plt.grid(True)
-- MAGIC plt.legend()
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The Gap suggests overfitting.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Same model, now with regularization

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import numpy as np
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC from xgboost import XGBClassifier
-- MAGIC from sklearn.metrics import roc_auc_score
-- MAGIC
-- MAGIC # Prepare datasets
-- MAGIC df_train = df[df["data_split"] == "train"].copy()
-- MAGIC df_val   = df[df["data_split"] == "val"].copy()
-- MAGIC df_test  = df[df["data_split"] == "test"].copy()
-- MAGIC
-- MAGIC X_train_full = df_train[feature_cols]
-- MAGIC y_train_full = df_train["churned"]
-- MAGIC
-- MAGIC X_val = df_val[feature_cols]
-- MAGIC y_val = df_val["churned"]
-- MAGIC
-- MAGIC X_test = df_test[feature_cols]
-- MAGIC y_test = df_test["churned"]
-- MAGIC
-- MAGIC # Define training sizes
-- MAGIC train_sizes = np.linspace(0.1, 1.0, 10)
-- MAGIC
-- MAGIC # Containers for both models
-- MAGIC train_scores_unreg, val_scores_unreg, test_scores_unreg = [], [], []
-- MAGIC train_scores_reg, val_scores_reg, test_scores_reg = [], [], []
-- MAGIC
-- MAGIC # Loop through training sizes
-- MAGIC for frac in train_sizes:
-- MAGIC     size = int(frac * len(X_train_full))
-- MAGIC     X_sample = X_train_full.iloc[:size]
-- MAGIC     y_sample = y_train_full.iloc[:size]
-- MAGIC     
-- MAGIC     # ---------- Unregularized Model ----------
-- MAGIC     model_unreg = XGBClassifier(
-- MAGIC         use_label_encoder=False,
-- MAGIC         eval_metric='logloss',
-- MAGIC         random_state=42
-- MAGIC     )
-- MAGIC     model_unreg.fit(X_sample, y_sample)
-- MAGIC
-- MAGIC     train_scores_unreg.append(roc_auc_score(y_sample, model_unreg.predict_proba(X_sample)[:, 1]))
-- MAGIC     val_scores_unreg.append(roc_auc_score(y_val, model_unreg.predict_proba(X_val)[:, 1]))
-- MAGIC     test_scores_unreg.append(roc_auc_score(y_test, model_unreg.predict_proba(X_test)[:, 1]))
-- MAGIC
-- MAGIC     # ---------- Your Tuned (Regularized) Model ----------
-- MAGIC     model_reg = XGBClassifier(
-- MAGIC         max_depth=3,
-- MAGIC         reg_alpha=1,
-- MAGIC         subsample=0.8,
-- MAGIC         colsample_bytree=0.8,
-- MAGIC         learning_rate=0.05,
-- MAGIC         n_estimators=300,
-- MAGIC         use_label_encoder=False,
-- MAGIC         eval_metric='logloss',
-- MAGIC         random_state=42
-- MAGIC     )
-- MAGIC     model_reg.fit(X_sample, y_sample)
-- MAGIC
-- MAGIC     train_scores_reg.append(roc_auc_score(y_sample, model_reg.predict_proba(X_sample)[:, 1]))
-- MAGIC     val_scores_reg.append(roc_auc_score(y_val, model_reg.predict_proba(X_val)[:, 1]))
-- MAGIC     test_scores_reg.append(roc_auc_score(y_test, model_reg.predict_proba(X_test)[:, 1]))
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Ensure full training dataset size is used
-- MAGIC full_train_size = len(X_train_full)
-- MAGIC train_sizes_scaled = train_sizes * full_train_size
-- MAGIC
-- MAGIC plt.figure(figsize=(12, 7))
-- MAGIC
-- MAGIC # Shared colors
-- MAGIC train_color = "#1f77b4"       # Blue
-- MAGIC val_color = "#ff7f0e"         # Orange
-- MAGIC test_color = "#2ca02c"        # Green
-- MAGIC
-- MAGIC # Baseline: Dotted lines
-- MAGIC plt.plot(train_sizes_scaled, train_scores_unreg, 'o--', color=train_color, label="Train ROC-AUC (Baseline)")
-- MAGIC plt.plot(train_sizes_scaled, val_scores_unreg, 'o--', color=val_color, label="Validation ROC-AUC (Baseline)")
-- MAGIC plt.plot(train_sizes_scaled, test_scores_unreg, 'o--', color=test_color, label="Test ROC-AUC (Baseline)")
-- MAGIC
-- MAGIC # Tuned Model: Solid lines
-- MAGIC plt.plot(train_sizes_scaled, train_scores_reg, 'o-', color=train_color, label="Train ROC-AUC (Tuned Model)")
-- MAGIC plt.plot(train_sizes_scaled, val_scores_reg, 'o-', color=val_color, label="Validation ROC-AUC (Tuned Model)")
-- MAGIC plt.plot(train_sizes_scaled, test_scores_reg, 'o-', color=test_color, label="Test ROC-AUC (Tuned Model)")
-- MAGIC
-- MAGIC # Styling
-- MAGIC plt.title("Learning Curve Comparison: XGBoost (Baseline vs Tuned Model)", fontsize=14)
-- MAGIC plt.xlabel("Training Set Size", fontsize=12)
-- MAGIC plt.ylabel("ROC-AUC Score", fontsize=12)
-- MAGIC plt.legend(loc='lower right', fontsize=10)
-- MAGIC plt.grid(True, linestyle='--', alpha=0.7)
-- MAGIC plt.xticks(fontsize=10)
-- MAGIC plt.yticks(fontsize=10)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Compute scaled training sizes
-- MAGIC full_train_size = len(X_train_full)
-- MAGIC train_sizes_scaled = train_sizes * full_train_size
-- MAGIC
-- MAGIC plt.figure(figsize=(13, 8))
-- MAGIC
-- MAGIC train_color = "#1f77b4"      # Blue
-- MAGIC val_color = "#ff7f0e"        # Orange
-- MAGIC test_color = "#d62728"       # Red
-- MAGIC
-- MAGIC # Baseline (dotted) - Thicker lines
-- MAGIC plt.plot(train_sizes_scaled, train_scores_unreg, 'o--', color=train_color,
-- MAGIC          label="Train ROC-AUC (Baseline)", linewidth=2.0, markersize=6)
-- MAGIC plt.plot(train_sizes_scaled, val_scores_unreg, 'o--', color=val_color,
-- MAGIC          label="Validation ROC-AUC (Baseline)", linewidth=2.0, markersize=6)
-- MAGIC plt.plot(train_sizes_scaled, test_scores_unreg, 'o--', color=test_color,
-- MAGIC          label="Test ROC-AUC (Baseline)", linewidth=2.0, markersize=6)
-- MAGIC
-- MAGIC # Tuned (solid) - Thicker lines
-- MAGIC plt.plot(train_sizes_scaled, train_scores_reg, 'o-', color=train_color,
-- MAGIC          label="Train ROC-AUC (Tuned Model)", linewidth=2.5, markersize=6)
-- MAGIC plt.plot(train_sizes_scaled, val_scores_reg, 'o-', color=val_color,
-- MAGIC          label="Validation ROC-AUC (Tuned Model)", linewidth=2.5, markersize=6)
-- MAGIC plt.plot(train_sizes_scaled, test_scores_reg, 'o-', color=test_color,
-- MAGIC          label="Test ROC-AUC (Tuned Model)", linewidth=2.5, markersize=6)
-- MAGIC
-- MAGIC # Annotate the generalization gap between Train and Test (Tuned Model)
-- MAGIC final_idx = -1
-- MAGIC final_train = train_scores_reg[final_idx]
-- MAGIC final_test = test_scores_reg[final_idx]
-- MAGIC final_size = train_sizes_scaled[final_idx]
-- MAGIC gap = final_train - final_test
-- MAGIC
-- MAGIC # Draw arrow and annotation
-- MAGIC plt.annotate(
-- MAGIC     f"Gap: {gap:.3f}",
-- MAGIC     xy=(final_size, final_test + (gap / 2)),
-- MAGIC     xytext=(final_size + 800, final_test + (gap / 2)),
-- MAGIC     arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5, linestyle='--'),
-- MAGIC     fontsize=12,
-- MAGIC     fontweight='bold',
-- MAGIC     ha='left',
-- MAGIC     va='center',
-- MAGIC     color='black'
-- MAGIC )
-- MAGIC
-- MAGIC # Final plot adjustments
-- MAGIC plt.title("Learning Curve Comparison: XGBoost (Baseline vs Tuned Model)", fontsize=16)
-- MAGIC plt.xlabel("Training Set Size", fontsize=13)
-- MAGIC plt.ylabel("ROC-AUC Score", fontsize=13)
-- MAGIC plt.xticks(fontsize=11)
-- MAGIC plt.yticks(fontsize=11)
-- MAGIC plt.grid(True, linestyle='--', alpha=0.7)
-- MAGIC
-- MAGIC # Move legend to gap between baseline train/test
-- MAGIC plt.legend(loc='center right', fontsize=11)
-- MAGIC
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Your tuned model sacrifices a little training performance to gain much better test and validation accuracy — this is exactly what good regularization does.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Interpretation of Model Outcome 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Feature Importance 

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import pandas as pd
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Get feature importance from trained model
-- MAGIC importance_df = pd.DataFrame({
-- MAGIC     'feature': model.get_booster().feature_names,
-- MAGIC     'importance': model.feature_importances_
-- MAGIC }).sort_values(by='importance', ascending=False)
-- MAGIC
-- MAGIC # Plot top features
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.barh(importance_df['feature'][:15][::-1], importance_df['importance'][:15][::-1], color='steelblue')
-- MAGIC plt.title("Top 15 Feature Importances (XGBoost)")
-- MAGIC plt.xlabel("Importance")
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Summary Stats by Churn Status

-- COMMAND ----------

-- MAGIC %py
-- MAGIC # Combine features and label
-- MAGIC df_features = df[df["data_split"] == "test"].copy()
-- MAGIC df_features["churn_prediction"] = model.predict(df_features[feature_cols])
-- MAGIC
-- MAGIC # Compare churned vs. retained customers
-- MAGIC churned = df_features[df_features["churned"] == 1]
-- MAGIC retained = df_features[df_features["churned"] == 0]
-- MAGIC
-- MAGIC # Example: mean values for selected behavioral traits
-- MAGIC selected_cols = feature_cols
-- MAGIC churn_summary = churned[selected_cols].mean()
-- MAGIC retained_summary = retained[selected_cols].mean()
-- MAGIC
-- MAGIC comparison_df = pd.DataFrame({
-- MAGIC     "Churned Avg": churn_summary,
-- MAGIC     "Retained Avg": retained_summary,
-- MAGIC     "Difference": churn_summary - retained_summary
-- MAGIC }).sort_values(by="Difference", key=abs, ascending=False)
-- MAGIC
-- MAGIC display(comparison_df)
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC visualized as bar chart

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC # Assuming comparison_df already exists (from previous step)
-- MAGIC top_n = 15
-- MAGIC top_diff = comparison_df.iloc[:top_n].sort_values(by='Difference', ascending=True)
-- MAGIC
-- MAGIC # Bar chart
-- MAGIC plt.figure(figsize=(10, 7))
-- MAGIC bars = plt.barh(top_diff.index, top_diff["Difference"], color='crimson')
-- MAGIC plt.axvline(0, color='gray', linewidth=0.8)
-- MAGIC plt.title("Top 15 Feature Differences: Churned vs. Retained Customers")
-- MAGIC plt.xlabel("Churned Avg − Retained Avg")
-- MAGIC plt.ylabel("Feature")
-- MAGIC plt.grid(axis='x', linestyle='--', alpha=0.5)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Creating Sub-groups of Churned customers (for pen portraits)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Silhoutte score analysis to get the most efficient number of k clusters

-- COMMAND ----------

-- MAGIC %py
-- MAGIC from sklearn.metrics import silhouette_score
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC from sklearn.cluster import KMeans
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC
-- MAGIC # Prepare data
-- MAGIC churned_df = df[df["data_split"] == "test"]
-- MAGIC churned_df = churned_df[churned_df["churned"] == 1].copy()
-- MAGIC X_churn = churned_df[feature_cols]
-- MAGIC
-- MAGIC # Scale
-- MAGIC scaler = StandardScaler()
-- MAGIC X_scaled = scaler.fit_transform(X_churn)
-- MAGIC
-- MAGIC # Try different values of k
-- MAGIC k_values = range(2, 10)
-- MAGIC silhouette_scores = []
-- MAGIC
-- MAGIC for k in k_values:
-- MAGIC     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
-- MAGIC     labels = kmeans.fit_predict(X_scaled)
-- MAGIC     score = silhouette_score(X_scaled, labels)
-- MAGIC     silhouette_scores.append(score)
-- MAGIC
-- MAGIC # Plot the silhouette scores
-- MAGIC plt.figure(figsize=(8, 5))
-- MAGIC plt.plot(k_values, silhouette_scores, marker='o')
-- MAGIC plt.title("Silhouette Score vs. Number of Clusters")
-- MAGIC plt.xlabel("Number of Clusters (k)")
-- MAGIC plt.ylabel("Average Silhouette Score")
-- MAGIC plt.grid(True)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC
-- MAGIC # Find optimal k
-- MAGIC optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
-- MAGIC print(f"Optimal number of clusters: {optimal_k}")
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC K=2 

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC from sklearn.cluster import KMeans
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC # Step 1: Filter to churned customers in test set
-- MAGIC churned_df = df[df["data_split"] == "test"]
-- MAGIC churned_df = churned_df[churned_df["churned"] == 1].copy()
-- MAGIC
-- MAGIC # Step 2: Select features used in model
-- MAGIC X_churn = churned_df[feature_cols]
-- MAGIC
-- MAGIC # Step 3: Scale features
-- MAGIC scaler = StandardScaler()
-- MAGIC X_scaled = scaler.fit_transform(X_churn)
-- MAGIC
-- MAGIC # Step 4: Run K-Means with 3 clusters (fix here)
-- MAGIC kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
-- MAGIC churned_df["cluster"] = kmeans.fit_predict(X_scaled)
-- MAGIC
-- MAGIC # Step 5: Compare cluster profiles
-- MAGIC cluster_summary = churned_df.groupby("cluster")[feature_cols].mean().T
-- MAGIC cluster_summary["max_diff"] = cluster_summary.max(axis=1) - cluster_summary.min(axis=1)
-- MAGIC
-- MAGIC # Sort by most varying features
-- MAGIC cluster_summary_sorted = cluster_summary.sort_values(by="max_diff", ascending=False).drop(columns="max_diff")
-- MAGIC
-- MAGIC display(cluster_summary_sorted)  # or print() if you're outside of a notebook
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC K = 3

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC # K=3
-- MAGIC kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
-- MAGIC churned_df["cluster_3"] = kmeans_3.fit_predict(X_scaled)
-- MAGIC
-- MAGIC cluster_summary_3 = churned_df.groupby("cluster_3")[feature_cols].mean().T
-- MAGIC cluster_summary_3["max_diff"] = cluster_summary_3.max(axis=1) - cluster_summary_3.min(axis=1)
-- MAGIC cluster_summary_3_sorted = cluster_summary_3.sort_values(by="max_diff", ascending=False).drop(columns="max_diff")
-- MAGIC
-- MAGIC display(cluster_summary_3_sorted)
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Visualizing Clusters

-- COMMAND ----------

-- MAGIC %py
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC from sklearn.decomposition import PCA
-- MAGIC
-- MAGIC # PCA to 2D for plotting
-- MAGIC pca_2d = PCA(n_components=2, random_state=42)
-- MAGIC X_pca_2d = pca_2d.fit_transform(X_scaled)
-- MAGIC
-- MAGIC # Map cluster labels to names
-- MAGIC label_map = {
-- MAGIC     0: "Dormant Deadweight",
-- MAGIC     1: "Recently Declining",
-- MAGIC     2: "High-Value Lapsed"
-- MAGIC }
-- MAGIC colors = ['red', 'orange', 'green']
-- MAGIC
-- MAGIC # Compute cluster centers in 2D PCA space
-- MAGIC cluster_col = "cluster_3" if "cluster_3" in churned_df.columns else "cluster_pca_3"
-- MAGIC
-- MAGIC plt.figure(figsize=(8, 6))
-- MAGIC
-- MAGIC for cluster_id in range(3):
-- MAGIC     mask = churned_df[cluster_col] == cluster_id
-- MAGIC     plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
-- MAGIC                 label=label_map[cluster_id], alpha=0.6, s=40, c=colors[cluster_id])
-- MAGIC
-- MAGIC     # Place text at cluster center
-- MAGIC     x_center = X_pca_2d[mask, 0].mean()
-- MAGIC     y_center = X_pca_2d[mask, 1].mean()
-- MAGIC     plt.text(x_center, y_center, label_map[cluster_id], fontsize=10, fontweight='bold', 
-- MAGIC              ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
-- MAGIC
-- MAGIC plt.title("2D PCA Cluster Visualization (Annotated)")
-- MAGIC plt.xlabel("Principal Component 1")
-- MAGIC plt.ylabel("Principal Component 2")
-- MAGIC plt.grid(True, linestyle="--", alpha=0.5)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Current, at risk customers assigned to clusters

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC # Filter customers predicted to churn (threshold = 0.5)
-- MAGIC predicted_churners = df_test[df_test["churn_probability"] >= 0.5].copy()
-- MAGIC
-- MAGIC # Extract their features
-- MAGIC X_predicted_churners = predicted_churners[feature_cols]
-- MAGIC
-- MAGIC # Scale features (reuse the same scaler you used before clustering)
-- MAGIC X_predicted_churners_scaled = scaler.transform(X_predicted_churners)
-- MAGIC
-- MAGIC # Predict cluster group using trained kmeans_3 model
-- MAGIC predicted_churners["cluster_group"] = kmeans_3.predict(X_predicted_churners_scaled)
-- MAGIC
-- MAGIC # Sort by churn likelihood
-- MAGIC top_100 = predicted_churners.sort_values(by="churn_probability", ascending=False).head(100)
-- MAGIC
-- MAGIC # Select columns to view
-- MAGIC output_df = top_100[["customer_id", "churn_probability", "cluster_group"]]
-- MAGIC
-- MAGIC # Display nicely in Databricks
-- MAGIC display(output_df)
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Output

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC from sklearn.cluster import KMeans
-- MAGIC
-- MAGIC # Get predicted churners
-- MAGIC predicted_churners = df_test[df_test["churn_probability"] >= 0.5].copy()
-- MAGIC
-- MAGIC # Only use feature columns for clustering
-- MAGIC X_churn = predicted_churners[feature_cols]
-- MAGIC X_scaled = StandardScaler().fit_transform(X_churn)
-- MAGIC
-- MAGIC # Fit K-Means on predicted churners 
-- MAGIC kmeans_predicted = KMeans(n_clusters=3, random_state=42, n_init=10)
-- MAGIC predicted_churners["cluster_group"] = kmeans_predicted.fit_predict(X_scaled)
-- MAGIC
-- MAGIC # Add human-readable labels
-- MAGIC pen_portraits = {
-- MAGIC     0: "Let Them Go",
-- MAGIC     1: "Recently Declining",
-- MAGIC     2: "High-Value Lapsed"
-- MAGIC }
-- MAGIC predicted_churners["pen_portrait"] = predicted_churners["cluster_group"].map(pen_portraits)
-- MAGIC

-- COMMAND ----------

-- MAGIC %py
-- MAGIC
-- MAGIC # Optional relabeling for clarity
-- MAGIC top_100["pen_portrait"] = top_100["pen_portrait"].replace("Dormant Deadweight", "Let them go")
-- MAGIC
-- MAGIC # Format table for display
-- MAGIC table = top_100[["customer_id", "churn_probability", "cluster_group", "pen_portrait"]]
-- MAGIC
-- MAGIC # Convert to styled HTML with scroll
-- MAGIC html = table.to_html(index=False, border=0, justify="center", classes="table table-striped", float_format="%.4f")
-- MAGIC
-- MAGIC styled_html = f"""
-- MAGIC <style>
-- MAGIC     .scrollable-table {{
-- MAGIC         max-height: 500px;
-- MAGIC         overflow-y: auto;
-- MAGIC         display: block;
-- MAGIC         font-family: Arial, sans-serif;
-- MAGIC     }}
-- MAGIC     table {{
-- MAGIC         border-collapse: collapse;
-- MAGIC         width: 100%;
-- MAGIC     }}
-- MAGIC     th, td {{
-- MAGIC         padding: 8px 12px;
-- MAGIC         text-align: center;
-- MAGIC         border-bottom: 1px solid #ddd;
-- MAGIC     }}
-- MAGIC     th {{
-- MAGIC         background-color: #f2f2f2;
-- MAGIC     }}
-- MAGIC </style>
-- MAGIC <div class="scrollable-table">
-- MAGIC {html}
-- MAGIC </div>
-- MAGIC """
-- MAGIC
-- MAGIC # Display in Databricks
-- MAGIC displayHTML(styled_html)
-- MAGIC