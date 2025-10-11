-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Data Base Initiation

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import urllib
-- MAGIC db_type = 'datas'
-- MAGIC data_id = 50
-- MAGIC exec(urllib.request.urlopen("https://docs.google.com/uc?export=download&id=1yYUXIPHgEj9AA-372uZiy4MFThW5Yc_1").read())
-- MAGIC
-- MAGIC import pandas as pd
-- MAGIC import matplotlib 
-- MAGIC import seaborn

-- COMMAND ----------

SHOW TABLES 

-- COMMAND ----------

-- Create temporary views for each table for experimenting without changing the form of orignial tables
CREATE OR REPLACE TEMP VIEW customers_view AS SELECT * FROM datas50.customers;
CREATE OR REPLACE TEMP VIEW products_view AS SELECT * FROM datas50.products;
CREATE OR REPLACE TEMP VIEW receipt_lines_view AS SELECT * FROM datas50.receipt_lines;
CREATE OR REPLACE TEMP VIEW receipts_view AS SELECT * FROM datas50.receipts;
CREATE OR REPLACE TEMP VIEW stores_view AS SELECT * FROM datas50.stores;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Individual Tables

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Creating individual tables to view and validate data 

-- COMMAND ----------

SELECT 
    customer_id, 
    first, 
    last, 
    dob
FROM 
    customers;


-- COMMAND ----------

SELECT 
    product_code, 
    product_details, 
    department_code, 
    department_name, 
    category_code, 
    category_details, 
    sub_category_code, 
    sub_category_details
FROM 
    products;


-- COMMAND ----------

SELECT 
    receipt_line_id, 
    receipt_id, 
    product_code, 
    qty, 
    value
FROM 
    receipt_lines;


-- COMMAND ----------

SELECT 
    receipt_id, 
    purchased_at, 
    customer_id, 
    store_code, 
    till_number
FROM 
    receipts;


-- COMMAND ----------

SELECT 
    store_code, 
    address, 
    postcode, 
    lat, 
    lng
FROM 
    stores;


-- COMMAND ----------

SELECT 
    region, 
    population
FROM 
    population_data;


-- COMMAND ----------

SELECT 
    region, 
    disposable_income
FROM 
    disposable_income_data;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Data Cleaning & Error Checking Data Set

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Value Error: 9999999
-- MAGIC - There are a group of 'values' showing up as '9999999', this must be an error in the data. We will explore the issue and resolve it 

-- COMMAND ----------

SELECT 
    r.receipt_id,
    rl.receipt_line_id,
    rl.product_code,
    rl.qty,
    rl.value,
    rl.qty * rl.value AS line_total
FROM receipts r
JOIN receipt_lines rl
    ON r.receipt_id = rl.receipt_id
WHERE r.customer_id = 18
ORDER BY line_total DESC;


-- COMMAND ----------

SELECT 
    receipt_id, product_code, qty, value
FROM receipt_lines
WHERE receipt_id IN (47150, 111416, 116, 101970, 63647, 59461)
ORDER BY receipt_id, product_code;


-- COMMAND ----------

SELECT 
    product_code, AVG(value) AS correct_value
FROM receipt_lines
WHERE value != 9999999
GROUP BY product_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC -- Here we take the lines the receipts that contained the value = 9999999 error and replaced the values with their correct values, as found on other receipts

-- COMMAND ----------

-- Here we take the lines the receipts that contained the value = 9999999 error and replaced the values with their correct values, as found on other receipts

UPDATE receipt_lines rl
SET rl.value = (
    SELECT AVG(rl2.value)
    FROM receipt_lines rl2
    WHERE rl2.product_code = rl.product_code
      AND rl2.value != 9999999
)
WHERE rl.receipt_id IN (47150, 111416, 116, 101970, 63647, 59461)
  AND rl.value = 9999999;


-- COMMAND ----------

SELECT 
    receipt_id, product_code, qty, value
FROM receipt_lines
WHERE receipt_id IN (47150, 111416, 116, 101970, 63647, 59461)
  AND value != 9999999;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Error Value: Weird '72' qty error

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Some  lines had strange reoccurances of "qty" = 72, so we investigated **this**

-- COMMAND ----------

--Some  lines had strange reoccurances of "qty" = 72, so we investigated this
SELECT 
    rl.receipt_id,
    rl.product_code,
    p.product_details,
    rl.qty,
    rl.value
FROM receipt_lines rl
JOIN products p
    ON rl.product_code = p.product_code
WHERE rl.qty = 72;

--Upon further investigation, it seems that this error happened on multiple receipts, all for the same product. I do not beleive this to be correct, but without a good 'product detail' there is not much we can do about it. Since the value is so low, maybe these customers did just buy a bunch of the items, so we will leave it as is.


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Error Check: High Value Products

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Some products seemed to be priced abnormally, so we will look into and fix any issues with pricing

-- COMMAND ----------

--Look into high value products to try to catch any discrepencies

SELECT 
    rl.receipt_id,
    rl.product_code,
    p.product_details,
    rl.qty,
    rl.value
FROM receipt_lines rl
JOIN products p
    ON rl.product_code = p.product_code
WHERE rl.value > 100;

--Everything looks normal enough to proceed, no pricing issues found upon further investigation


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Error Value: Null Quantities

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Some 'qty' entries are appearing as null. We will investigate an fix this issue.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC --It looks like all of the 'null' outputs are quiet random, and that the only commonality is that the some of the null's happened on certain matching receipts. We will fill 'null' with 1, as this will help complete our data set and should not cause any issues (and since 1 is likely the correct qty for most of these items since the value of the items is so small) This will help complete our dataset without compomising anything.

-- COMMAND ----------

SELECT 
    rl.receipt_id,
    rl.product_code,
    rl.qty,
    rl.value
FROM receipt_lines rl
WHERE rl.qty IS NULL OR rl.value IS NULL
ORDER BY rl.receipt_id;


-- COMMAND ----------

SELECT 
    rl.receipt_id,
    rl.product_code,
    rl.qty,
    p.product_details
FROM receipt_lines rl
JOIN products p
    ON rl.product_code = p.product_code
WHERE rl.qty IS NULL
ORDER BY rl.receipt_id;

--It looks like all of the 'null' outputs are quiet random, and that the only commonality is that the null's happened on certain receipts. We will fill 'null' with 1, as this will help complete our data set and should not cause any issues (and since 1 is likely the correct qty for most of these items)

-- COMMAND ----------

UPDATE receipt_lines
SET qty = 1
WHERE qty IS NULL;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Error Value: Negative Cost Line Items

-- COMMAND ----------

-- MAGIC %md
-- MAGIC --Here we are checking 'negative value items' to ensure there are no errors in the data. Negative line items seems like a strange occurance

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Upon further investigation, the negative line items do seem to be purposeful. One of the line items is under 'lottery' which indicates that when the store pays someone, it appears as a negative line item on the receipts. Because of this, we should keep the negative values since they are all likely normal operations from the stores.

-- COMMAND ----------

--Here we are checking 'negative value items' to ensure there are no errors in the data

SELECT 
    rl.product_code,
    p.product_details,
    SUM(rl.value) AS total_value_spent
FROM receipt_lines_view rl
JOIN products p
    ON rl.product_code = p.product_code
GROUP BY rl.product_code, p.product_details
HAVING SUM(rl.value) < 0;


--It seems that negative values must be lottery winnings and returns. It makes sense to keep these items as they are

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Creating New Tables With Outside Data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Here we are creating tables for the outside data sources that were found and may potentially be used in our analysis

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **New Tables:**
-- MAGIC - population_data
-- MAGIC - disposable_income_data
-- MAGIC - grocery_spending_data

-- COMMAND ----------

CREATE or REPLACE TABLE population_data (
    region STRING,
    population INT
);

INSERT INTO population_data VALUES
('Nottingham', 329276),
('Birmingham', 1166049),
('London', 8945309);


-- COMMAND ----------

CREATE OR REPLACE TABLE disposable_income_data (
    region STRING,
    disposable_income INT
);

INSERT INTO disposable_income_data VALUES
('Nottingham', 15434),
('Birmingham', 16950),
('London', 32330);


-- COMMAND ----------

CREATE OR REPLACE TABLE grocery_spending_data (
    region STRING,
    grocery_spending DECIMAL(10, 2)
);

INSERT INTO grocery_spending_data VALUES
('Nottingham', 184.66),
('Birmingham', 157.34),
('London', 153.12);


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Initial Exploration of Tables: Basic Table Visualizaiton
-- MAGIC - In this section we look at the data within each table, without joining tables
-- MAGIC - Our goal here is to not yet start making KPI's, but to get a better understanding of the data to make a more informed decision when it comes to choosing which KPI's would be the most effective

-- COMMAND ----------

SELECT
    YEAR(r.purchased_at) AS year,
    r.store_code,
    COUNT(DISTINCT r.customer_id) AS total_customers
FROM datas50.receipts r
WHERE r.customer_id IS NOT NULL AND r.customer_id != -1
GROUP BY YEAR(r.purchased_at), r.store_code
ORDER BY year ASC, r.store_code ASC;


-- COMMAND ----------

SELECT * FROM customers_view LIMIT 100;




-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Customers Table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of Unique customer ID's

-- COMMAND ----------

SELECT 
    r.store_code, 
    COUNT(DISTINCT rl.product_code) AS product_count
FROM 
    datas50.receipts r
JOIN 
    datas50.receipt_lines rl 
ON 
    r.receipt_id = rl.receipt_id
GROUP BY 
    r.store_code
ORDER BY 
    r.store_code;




-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of repeat customer ID's

-- COMMAND ----------

SELECT 
    first,
    last,
    COUNT(*) AS occurrences
FROM customers
GROUP BY first, last
HAVING COUNT(*) > 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Products Table

-- COMMAND ----------

SELECT * FROM products_view LIMIT 100;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique products

-- COMMAND ----------

SELECT COUNT(DISTINCT product_code) AS unique_product_count FROM products_view;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique products in each department

-- COMMAND ----------

SELECT department_code, COUNT(DISTINCT product_code) AS unique_products_in_department
FROM products_view
GROUP BY department_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique departments

-- COMMAND ----------

SELECT COUNT(DISTINCT department_code) AS unique_department_count FROM products_view;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique categories

-- COMMAND ----------

SELECT COUNT(DISTINCT category_code) AS unique_category_count FROM products_view;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC Here we create a graph to get a better look at the number of unique products in each department

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import seaborn as sns
-- MAGIC
-- MAGIC df = spark.sql("""
-- MAGIC SELECT department_code, COUNT(DISTINCT product_code) AS unique_products_in_department
-- MAGIC FROM products_view
-- MAGIC GROUP BY department_code
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.barplot(x="department_code", y="unique_products_in_department", data=df)
-- MAGIC plt.title("Unique Products in Each Department")
-- MAGIC plt.xticks(rotation=45)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Receipt Lines Table

-- COMMAND ----------

SELECT * FROM receipt_lines_view LIMIT 100;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique receipt line IDs:

-- COMMAND ----------

SELECT COUNT(DISTINCT receipt_line_id) AS unique_receipt_line_count FROM receipt_lines_view;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique receipt IDs

-- COMMAND ----------

SELECT COUNT(DISTINCT receipt_id) AS unique_receipt_count FROM receipt_lines_view;


-- COMMAND ----------

SELECT COUNT(customer_id) - COUNT(DISTINCT customer_id) AS repeat_customer_count FROM receipts;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Count of each product with quantities

-- COMMAND ----------

SELECT 
    product_code, 
    SUM(qty) AS total_quantity_sold 
FROM receipt_lines_view
GROUP BY product_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Total value spent on each product

-- COMMAND ----------

SELECT 
    product_code, 
    SUM(value) AS total_value_spent 
FROM receipt_lines_view
GROUP BY product_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Here we create a visualization to see the top 10 product purchased within the store

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = spark.sql("""
-- MAGIC SELECT 
-- MAGIC     product_code, 
-- MAGIC     SUM(value) AS total_value_spent 
-- MAGIC FROM receipt_lines_view
-- MAGIC GROUP BY product_code
-- MAGIC ORDER BY total_value_spent DESC
-- MAGIC LIMIT 10
-- MAGIC """).toPandas()
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.barplot(x="product_code", y="total_value_spent", data=df)
-- MAGIC plt.title("Top 10 Products by Total Value Spent")
-- MAGIC plt.xticks(rotation=45)
-- MAGIC plt.tight_layout()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Receipts Table

-- COMMAND ----------

SELECT * FROM receipts_view LIMIT 100;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique receipts from each customer:

-- COMMAND ----------

SELECT 
    customer_id, 
    COUNT(DISTINCT receipt_id) AS unique_receipts_per_customer
FROM receipts_view
GROUP BY customer_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of unique shoppers in each store

-- COMMAND ----------

SELECT 
    store_id, 
    COUNT(DISTINCT customer_id) AS unique_shoppers_per_store
FROM receipts_view
GROUP BY store_id;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Number of receipts in each store:

-- COMMAND ----------

SELECT 
    store_code, 
    COUNT(receipt_id) AS receipt_count_per_store
FROM receipts_view
GROUP BY store_code;


-- COMMAND ----------

SELECT 
    MIN(purchased_at) AS min_date, 
    MAX(purchased_at) AS max_date,
    DATEDIFF(MAX(purchased_at), MIN(purchased_at)) AS span_days
FROM receipts;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Customers shopping in multiple stores

-- COMMAND ----------

SELECT customer_id
FROM receipts_view
GROUP BY customer_id
HAVING COUNT(DISTINCT store_code) > 1;


-- COMMAND ----------

SELECT * FROM stores_view LIMIT 100;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #KPI Section
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Here we start exploring some potential KPI's

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Customers who visited the most times, by first and last name

-- COMMAND ----------

SELECT 
    c.customer_id,
    c.first,
    c.last,
    COUNT(DISTINCT r.receipt_id) AS total_visits,
    SUM(rl.value) AS total_value_spent
FROM customers c
JOIN receipts r
    ON c.customer_id = r.customer_id
JOIN receipt_lines rl
    ON r.receipt_id = rl.receipt_id
GROUP BY c.customer_id, c.first, c.last
ORDER BY total_value_spent DESC;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Active Customers
-- MAGIC - Number of unique customers who made a purchase in a given time period.
-- MAGIC - Bar Chart: Store code on the x-axis, active customers on the y-axis.

-- COMMAND ----------

SELECT 
    r. store_code, 
    COUNT(DISTINCT r.customer_id) AS active_customers
FROM 
    receipts r
WHERE 
    r.purchased_at BETWEEN '2021-01-01' AND '2021-12-31' -- Adjust the time range
GROUP BY 
    r.store_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Active Repeat Customers
-- MAGIC - Customers who made at least two purchases in a given time period.
-- MAGIC - Bar Chart: Store code on the x-axis, active repeat customers on the y-axis.

-- COMMAND ----------

SELECT 
    sub.store_code, 
    COUNT(DISTINCT sub.customer_id) AS active_repeat_customers
FROM (
    SELECT 
        customer_id, 
        store_code, 
        COUNT(*) AS purchase_count
    FROM 
        receipts
    WHERE 
        purchased_at BETWEEN '2021-01-01' AND '2021-12-31' -- Adjust the time range
    GROUP BY 
        customer_id, store_code
) sub
WHERE 
    sub.purchase_count > 1
GROUP BY 
    sub.store_code;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Customer Retention Rate
-- MAGIC - Percentage of customers who made a repeat purchase within a given time frame.
-- MAGIC - Bar Chart or Line Chart: Retention rate (%) on the y-axis, store code or region on the x-axis.

-- COMMAND ----------

WITH first_purchase AS (
    SELECT customer_id, store_code, MIN(purchased_at) AS first_purchase_date
    FROM receipts
    GROUP BY customer_id, store_code
),
repeat_customers AS (
    SELECT DISTINCT r.customer_id, r.store_code
    FROM receipts r
    JOIN first_purchase fp
    ON r.customer_id = fp.customer_id AND r.store_code = fp.store_code
    WHERE r.purchased_at > fp.first_purchase_date
)
SELECT 
    fp.store_code, 
    COUNT(rc.customer_id) * 100.0 / COUNT(fp.customer_id) AS retention_rate
FROM 
    first_purchase fp
LEFT JOIN 
    repeat_customers rc
ON 
    fp.customer_id = rc.customer_id AND fp.store_code = rc.store_code
GROUP BY 
    fp.store_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Customer Lifetime Value
-- MAGIC - Average total revenue generated by a customer over their lifetime.
-- MAGIC - Bar Chart or Heatmap: Store code on the x-axis, CLV on the y-axis.
-- MAGIC

-- COMMAND ----------

SELECT 
    sub.store_code, 
    AVG(sub.total_spent) AS avg_customer_lifetime_value
FROM (
    SELECT 
        r.customer_id, 
        r.store_code, 
        SUM(rl.value) AS total_spent
    FROM 
        receipts r
    JOIN 
        receipt_lines rl 
    ON 
        r.receipt_id = rl.receipt_id
    GROUP BY 
        r.customer_id, r.store_code
) sub
GROUP BY 
    sub.store_code;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Churn Rate
-- MAGIC - Definition: Percentage of customers who did not make a repeat purchase within a set time.
-- MAGIC - Line Chart or Bar Chart: Churn rate (%) on the y-axis, store code or region on the x-axis.

-- COMMAND ----------

WITH first_last_purchase AS (
    SELECT 
        customer_id, 
        store_code, 
        MIN(purchased_at) AS first_purchase_date, 
        MAX(purchased_at) AS last_purchase_date
    FROM 
        receipts
    GROUP BY 
        customer_id, store_code
)
SELECT 
    store_code, 
    COUNT(CASE WHEN last_purchase_date < '2021-12-31' THEN customer_id END) * 100.0 / COUNT(customer_id) AS churn_rate
FROM 
    first_last_purchase
GROUP BY 
    store_code;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## High Value and Low Value Customers
-- MAGIC - Definition: High-value customers are those whose total spending is above a given threshold, while low-value customers are below it.
-- MAGIC - Stacked Bar Chart: High-value and low-value customer counts for each store.
-- MAGIC

-- COMMAND ----------

SELECT 
    r.store_code,
    -- High value customers (spent >= 50 on a single receipt)
    COUNT(DISTINCT CASE WHEN total_spent >= 50 THEN r.customer_id END) AS high_value_customers,
    -- Low value customers (spent < 50 on all receipts)
    COUNT(DISTINCT CASE WHEN total_spent < 50 THEN r.customer_id END) AS low_value_customers
FROM 
    receipts r
JOIN 
    (
        SELECT 
            r.receipt_id,
            r.customer_id,
            SUM(rl.value) AS total_spent
        FROM 
            receipts r
        JOIN 
            receipt_lines rl 
        ON 
            r.receipt_id = rl.receipt_id
        GROUP BY 
            r.receipt_id, r.customer_id
    ) sub 
ON 
    r.customer_id = sub.customer_id
GROUP BY 
    r.store_code;





-- COMMAND ----------

SELECT 
    r.receipt_id,
    r.customer_id,
    r.store_code,
    rl.receipt_line_id,
    rl.product_code,
    rl.value AS line_item_value
FROM 
    receipts r
JOIN 
    receipt_lines rl 
ON 
    r.receipt_id = rl.receipt_id
WHERE 
    r.store_code = 2  -- Filter for Store 2
ORDER BY 
    r.receipt_id
LIMIT 100;  -- Limit to 100 receipts



-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Import necessary libraries
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC # Assuming you are using a SparkSession to query your data.
-- MAGIC # Let's execute a SQL query to fetch receipt_id and value from receipt_lines
-- MAGIC
-- MAGIC query = """
-- MAGIC SELECT receipt_id, SUM(value) as total_value
-- MAGIC FROM receipt_lines
-- MAGIC GROUP BY receipt_id
-- MAGIC """
-- MAGIC
-- MAGIC # Execute the query to get the data into a Spark DataFrame
-- MAGIC df = spark.sql(query)
-- MAGIC
-- MAGIC # Convert the Spark DataFrame to a Pandas DataFrame for further analysis
-- MAGIC pandas_df = df.toPandas()
-- MAGIC
-- MAGIC # Display the result
-- MAGIC print(pandas_df)
-- MAGIC
-- MAGIC # If you want to visualize the data
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC plt.bar(pandas_df['receipt_id'], pandas_df['total_value'], color='skyblue', edgecolor='black')
-- MAGIC plt.title('Total Value per Receipt ID')
-- MAGIC plt.xlabel('Receipt ID')
-- MAGIC plt.ylabel('Total Value ($)')
-- MAGIC plt.grid(True)
-- MAGIC plt.show()
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Advanced KPI's

-- COMMAND ----------

-- MAGIC %md
-- MAGIC These KPI's are more advanced, and most use the outside data sources 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Regional Market Penetration
-- MAGIC - Definition: Percentage of the region’s population that are active customers.
-- MAGIC - Purpose: Evaluates how well each store captures its local market.
-- MAGIC - Market Saturation:
-- MAGIC Estimate potential growth by comparing the store’s market penetration to the regional population. Regions with low penetration but high disposable income indicate untapped potential.
-- MAGIC
-- MAGIC
-- MAGIC ***_Requires Regional Population Metrics_***

-- COMMAND ----------

WITH store_regions AS (
    SELECT 
        CASE 
            WHEN store_code = 0 THEN 'Nottingham'
            WHEN store_code = 1 THEN 'Birmingham'
            WHEN store_code IN (2, 3) THEN 'London'
        END AS region,
        store_code
    FROM stores
)
SELECT 
    sr.region, 
    COUNT(DISTINCT r.customer_id) AS active_customers,
    pd.population,
    (COUNT(DISTINCT r.customer_id) * 100.0 / pd.population) AS market_penetration_percentage
FROM 
    store_regions sr
JOIN 
    receipts r ON sr.store_code = r.store_code
JOIN 
    population_data pd ON sr.region = pd.region
GROUP BY 
    sr.region, pd.population;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Spending vs. Regional Income Ratio
-- MAGIC - Definition: Average customer spending as a percentage of regional gross disposable income.
-- MAGIC - Purpose: Shows whether customer spending aligns with regional income levels.
-- MAGIC
-- MAGIC **_Requires Regional Income Data_**

-- COMMAND ----------

WITH store_regions AS (
    SELECT 
        CASE 
            WHEN store_code = 0 THEN 'Nottingham'
            WHEN store_code = 1 THEN 'Birmingham'
            WHEN store_code IN (2, 3) THEN 'London'
        END AS region,
        store_code
    FROM stores
),
customer_spending AS (
    SELECT 
        r.store_code,
        r.customer_id,
        SUM(rl.value) AS total_spent
    FROM 
        receipts r
    JOIN 
        receipt_lines rl ON r.receipt_id = rl.receipt_id
    GROUP BY 
        r.store_code, r.customer_id
)
SELECT 
    sr.region, 
    AVG(cs.total_spent) AS avg_customer_spending,
    di.disposable_income,
    (AVG(cs.total_spent) / di.disposable_income) AS spending_vs_income_ratio
FROM 
    store_regions sr
JOIN 
    customer_spending cs ON sr.store_code = cs.store_code
JOIN 
    disposable_income_data di ON sr.region = di.region
GROUP BY 
    sr.region, di.disposable_income;



-- COMMAND ----------

-- MAGIC %md
-- MAGIC Checking average customer spending in the regions to verify results above

-- COMMAND ----------

SELECT 
    customer_id,
    SUM(rl.value) AS total_spent
FROM 
    receipts r
JOIN 
    receipt_lines rl 
ON 
    r.receipt_id = rl.receipt_id
WHERE
    r.store_code = 0  -- For Birmingham
GROUP BY 
    customer_id
ORDER BY 
    total_spent DESC
LIMIT 10000;  -- Adjust this number as needed to get a sense of spending


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Checking for high frequency of low spending to verify above findings

-- COMMAND ----------

SELECT 
    r.customer_id,
    COUNT(DISTINCT r.receipt_id) AS visit_count,
    AVG(rl.value) AS avg_spent_per_visit
FROM 
    receipts r
JOIN 
    receipt_lines rl 
ON 
    r.receipt_id = rl.receipt_id
WHERE
    r.store_code = 1  -- For Birmingham
GROUP BY 
    r.customer_id
HAVING
    COUNT(DISTINCT r.receipt_id) > 1  -- Focus on repeat customers
ORDER BY 
    avg_spent_per_visit DESC
LIMIT 100;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC High vs low spend customers

-- COMMAND ----------

SELECT 
    customer_id,
    SUM(rl.value) AS total_spent,
    COUNT(DISTINCT r.receipt_id) AS num_receipts
FROM 
    receipts r
JOIN 
    receipt_lines rl 
ON 
    r.receipt_id = rl.receipt_id
WHERE
    r.store_code = 1  -- For Birmingham
GROUP BY 
    customer_id
ORDER BY 
    total_spent DESC
LIMIT 100;  -- Adjust to view top spenders


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Comparing product mix per store

-- COMMAND ----------

SELECT 
    rl.product_code,
    SUM(rl.value) AS total_value
FROM 
    receipt_lines rl
JOIN 
    receipts r
ON 
    r.receipt_id = rl.receipt_id
WHERE
    r.store_code = 1  -- For Birmingham
GROUP BY 
    rl.product_code
ORDER BY 
    total_value DESC
LIMIT 20;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Visualization of KPI's

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Retention Rate & Retention Rate Over Time

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd

-- COMMAND ----------


-- Calculate retention rate and create a view
CREATE OR REPLACE TEMP VIEW retention_rate_data AS
WITH first_purchase AS (
    SELECT customer_id, store_code, MIN(purchased_at) AS first_purchase_date
    FROM receipts
    GROUP BY customer_id, store_code
),
repeat_customers AS (
    SELECT DISTINCT r.customer_id, r.store_code
    FROM receipts r
    JOIN first_purchase fp
    ON r.customer_id = fp.customer_id AND r.store_code = fp.store_code
    WHERE r.purchased_at > fp.first_purchase_date
)
SELECT 
    fp.store_code, 
    COUNT(rc.customer_id) * 100.0 / COUNT(fp.customer_id) AS retention_rate
FROM 
    first_purchase fp
LEFT JOIN 
    repeat_customers rc
ON 
    fp.customer_id = rc.customer_id AND fp.store_code = rc.store_code
GROUP BY 
    fp.store_code;



-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Import necessary libraries
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC # Query the retention rate data from the temporary view
-- MAGIC retention_rate_df = spark.sql("SELECT * FROM retention_rate_data").toPandas()
-- MAGIC
-- MAGIC # Display the DataFrame to verify
-- MAGIC print(retention_rate_df)
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import seaborn as sns
-- MAGIC
-- MAGIC # Set up the plot
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.barplot(data=retention_rate_df, x="store_code", y="retention_rate", palette="viridis")
-- MAGIC
-- MAGIC # Add labels and title
-- MAGIC plt.xlabel("Store Code")
-- MAGIC plt.ylabel("Retention Rate (%)")
-- MAGIC plt.title("Customer Retention Rate by Store")
-- MAGIC plt.ylim(0, 100)
-- MAGIC
-- MAGIC # Display the plot
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Retention Rate Over Time

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Church Rate

-- COMMAND ----------

-- Calculate churn rate and create a view
CREATE OR REPLACE TEMP VIEW churn_rate_data AS
WITH first_last_purchase AS (
    SELECT 
        customer_id, 
        store_code, 
        MIN(purchased_at) AS first_purchase_date, 
        MAX(purchased_at) AS last_purchase_date
    FROM 
        receipts
    GROUP BY 
        customer_id, store_code
)
SELECT 
    store_code, 
    COUNT(CASE WHEN last_purchase_date < '2021-12-31' THEN customer_id END) * 100.0 / COUNT(customer_id) AS churn_rate
FROM 
    first_last_purchase
GROUP BY 
    store_code;


-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Query churn rate data
-- MAGIC churn_rate_df = spark.sql("SELECT * FROM churn_rate_data").toPandas()
-- MAGIC
-- MAGIC # Display the DataFrame to verify
-- MAGIC print(churn_rate_df)
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import seaborn as sns
-- MAGIC
-- MAGIC # Set up the plot
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC sns.barplot(data=churn_rate_df, x="store_code", y="churn_rate", palette="coolwarm")
-- MAGIC
-- MAGIC # Add labels and title
-- MAGIC plt.xlabel("Store Code")
-- MAGIC plt.ylabel("Churn Rate (%)")
-- MAGIC plt.title("Customer Churn Rate by Store")
-- MAGIC plt.ylim(0, 100)
-- MAGIC
-- MAGIC # Display the plot
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- Churn rate over time
CREATE OR REPLACE TEMP VIEW churn_rate_over_time AS
WITH first_last_purchase AS (
    SELECT 
        customer_id, 
        store_code, 
        MIN(purchased_at) AS first_purchase_date, 
        MAX(purchased_at) AS last_purchase_date
    FROM 
        receipts
    GROUP BY 
        customer_id, store_code
)
SELECT 
    store_code, 
    MONTH(last_purchase_date) AS last_purchase_month,
    COUNT(CASE WHEN last_purchase_date < '2021-12-31' THEN customer_id END) * 100.0 / COUNT(customer_id) AS churn_rate
FROM 
    first_last_purchase
GROUP BY 
    store_code, MONTH(last_purchase_date);


-- COMMAND ----------

-- MAGIC %python
-- MAGIC churn_rate_time_df = spark.sql("SELECT * FROM churn_rate_over_time").toPandas()
-- MAGIC
-- MAGIC # Line plot for churn rate over time
-- MAGIC plt.figure(figsize=(12, 6))
-- MAGIC sns.lineplot(data=churn_rate_time_df, x="last_purchase_month", y="churn_rate", hue="store_code", marker="o")
-- MAGIC
-- MAGIC # Add labels and title
-- MAGIC plt.xlabel("Month")
-- MAGIC plt.ylabel("Churn Rate (%)")
-- MAGIC plt.title("Customer Churn Rate Over Time")
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##High Spend Customers vs. Low Spend Customers

-- COMMAND ----------

WITH customer_total_spending AS (
    SELECT 
        r.store_code,
        r.customer_id,
        SUM(rl.value) AS total_spent
    FROM 
        receipts r
    JOIN 
        receipt_lines rl ON r.receipt_id = rl.receipt_id
    GROUP BY 
        r.store_code, r.customer_id
)
SELECT 
    store_code,
    COUNT(CASE WHEN total_spent >= 50 THEN customer_id END) AS high_value_customers,
    COUNT(CASE WHEN total_spent < 50 THEN customer_id END) AS low_value_customers
FROM 
    customer_total_spending
GROUP BY 
    store_code;



-- COMMAND ----------

-- MAGIC %python
-- MAGIC high_low_value_df = spark.sql("""
-- MAGIC WITH customer_total_spending AS (
-- MAGIC     SELECT 
-- MAGIC         r.store_code,
-- MAGIC         r.customer_id,
-- MAGIC         SUM(rl.value) AS total_spent
-- MAGIC     FROM 
-- MAGIC         receipts r
-- MAGIC     JOIN 
-- MAGIC         receipt_lines rl ON r.receipt_id = rl.receipt_id
-- MAGIC     GROUP BY 
-- MAGIC         r.store_code, r.customer_id
-- MAGIC )
-- MAGIC SELECT 
-- MAGIC     store_code,
-- MAGIC     COUNT(CASE WHEN total_spent >= 50 THEN customer_id END) AS high_value_customers,
-- MAGIC     COUNT(CASE WHEN total_spent < 50 THEN customer_id END) AS low_value_customers
-- MAGIC FROM 
-- MAGIC     customer_total_spending
-- MAGIC GROUP BY 
-- MAGIC     store_code
-- MAGIC """).toPandas()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Regional Market Penetration

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC
-- MAGIC data = {
-- MAGIC     'region': ['Birmingham', 'London', 'Nottingham'],
-- MAGIC     'active_customers': [3969, 3750, 2383],
-- MAGIC     'population': [1166049, 8945309, 329276],
-- MAGIC     'market_penetration_percentage': [0.340380207007, 0.041921413782, 0.723708985775]
-- MAGIC }
-- MAGIC
-- MAGIC df = pd.DataFrame(data)
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC
-- MAGIC # Set the figure size
-- MAGIC plt.figure(figsize=(10, 6))
-- MAGIC
-- MAGIC # Bar chart for market penetration percentage
-- MAGIC plt.bar(df['region'], df['market_penetration_percentage'], color=['blue', 'orange', 'green'])
-- MAGIC
-- MAGIC # Adding labels and title
-- MAGIC plt.xlabel('Region')
-- MAGIC plt.ylabel('Market Penetration Percentage (%)')
-- MAGIC plt.title('Regional Market Penetration')
-- MAGIC
-- MAGIC # Display percentage values on top of the bars
-- MAGIC for i, v in enumerate(df['market_penetration_percentage']):
-- MAGIC     plt.text(i, v + 0.02, f"{v:.2f}%", ha='center', va='bottom')
-- MAGIC
-- MAGIC # Show plot
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #Additional Visualization for Presentation of Data
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC # Query data with aggregation
-- MAGIC revenue_query = """
-- MAGIC SELECT 
-- MAGIC     DATE_TRUNC('month', r.purchased_at) AS month, 
-- MAGIC     r.store_code, 
-- MAGIC     SUM(rl.value) AS revenue
-- MAGIC FROM datas50.receipts r
-- MAGIC JOIN datas50.receipt_lines rl ON r.receipt_id = rl.receipt_id
-- MAGIC GROUP BY DATE_TRUNC('month', r.purchased_at), r.store_code
-- MAGIC ORDER BY month
-- MAGIC """
-- MAGIC revenue_data = spark.sql(revenue_query).toPandas()
-- MAGIC
-- MAGIC # Convert to datetime
-- MAGIC revenue_data['month'] = pd.to_datetime(revenue_data['month'])
-- MAGIC
-- MAGIC # Plot
-- MAGIC plt.figure(figsize=(12, 6))
-- MAGIC for store, group in revenue_data.groupby('store_code'):
-- MAGIC     plt.plot(group['month'], group['revenue'], label=f'Store {store}')
-- MAGIC
-- MAGIC plt.title('Monthly Revenue Trends Over Time by Store')
-- MAGIC plt.xlabel('Month')
-- MAGIC plt.ylabel('Revenue')
-- MAGIC plt.legend()
-- MAGIC plt.grid()
-- MAGIC plt.show()
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC
-- MAGIC # Query data with monthly aggregation
-- MAGIC monthly_customers_query = """
-- MAGIC SELECT 
-- MAGIC     DATE_TRUNC('month', r.purchased_at) AS month,
-- MAGIC     r.store_code,
-- MAGIC     COUNT(DISTINCT r.customer_id) AS customer_count
-- MAGIC FROM datas50.receipts r
-- MAGIC WHERE r.customer_id IS NOT NULL
-- MAGIC GROUP BY DATE_TRUNC('month', r.purchased_at), r.store_code
-- MAGIC ORDER BY month
-- MAGIC """
-- MAGIC monthly_customers_data = spark.sql(monthly_customers_query).toPandas()
-- MAGIC
-- MAGIC # Convert to datetime
-- MAGIC monthly_customers_data['month'] = pd.to_datetime(monthly_customers_data['month'])
-- MAGIC
-- MAGIC # Sort the data by store and month
-- MAGIC monthly_customers_data = monthly_customers_data.sort_values(['store_code', 'month'])
-- MAGIC
-- MAGIC # Apply a rolling average for smoothing
-- MAGIC monthly_customers_data['smoothed_customer_count'] = (
-- MAGIC     monthly_customers_data.groupby('store_code')['customer_count']
-- MAGIC     .transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
-- MAGIC )
-- MAGIC
-- MAGIC # Plot
-- MAGIC plt.figure(figsize=(12, 6))
-- MAGIC for store, group in monthly_customers_data.groupby('store_code'):
-- MAGIC     plt.plot(group['month'], group['smoothed_customer_count'], label=f'Store {store}')
-- MAGIC
-- MAGIC plt.title('Monthly Customer Count Over Time by Store (Smoothed)')
-- MAGIC plt.xlabel('Month')
-- MAGIC plt.ylabel('Number of Customers')
-- MAGIC plt.legend()
-- MAGIC plt.grid()
-- MAGIC plt.show()
-- MAGIC
-- MAGIC
