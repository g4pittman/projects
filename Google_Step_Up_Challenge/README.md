# 📊 Google Step Up Challenge — Gemini Marketing Campaign Strategy

<p align="center">
  <img src="https://github.com/g4pittman/projects/blob/main/Google_Step_Up_Challenge/PPT_Intro_Slide.png?raw=true" width="800"/>
</p>

## Overview
Developed a data-driven **$10M digital marketing strategy** to launch Google's Gemini Pro across **3 international markets** (UK, Germany, Saudi Arabia, Egypt) targeting university students over an 8-week campaign period.

This project demonstrates professional-level skills in **statistical analysis**, **marketing analytics**, **budget optimization**, and **data-driven decision making** — combining Python programming, business intelligence, and strategic thinking.

---

## 📑 Full Strategy Presentation

**[Download Complete Slide Deck](./Gemini_Campaign_Strategy.pptx)** (PowerPoint)

---

## 🎯 Challenge Objectives

- **Reach:** Efficiently target university students across multiple markets
- **Drive Consideration:** Increase intent to use Gemini Pro
- **Boost Sign-ups:** Maximize conversions within budget constraints
- **Budget:** $10M USD allocated strategically across markets and channels

---

## 🔧 Technical Approach & Skills Demonstrated

### 1. Statistical Analysis (Python)
- **Z-Test for Proportions:** Calculated statistical significance of brand lift studies using `statsmodels.stats.proportion.proportions_ztest`
- **Hypothesis Testing:** Filtered campaign results to only include statistically significant outcomes (p < 0.05)
- **Sample Size Analysis:** Evaluated control vs. exposed groups across multiple campaigns
```python
from statsmodels.stats.proportion import proportions_ztest

# Statistical significance testing
counts = [control_consideration, exposed_consideration]
sample_size = [control_responses, exposed_responses]
stat, pval = proportions_ztest(counts, sample_size)
```

### 2. Data Engineering & Transformation
- **Multi-table Joins:** Merged historic campaign data with brand lift studies using composite keys (Campaign_Name + Market + Channel)
- **Python (Pandas):** `merge()` operations for left joins across datasets
- **Excel:** VLOOKUP functions and join key creation for manual analysis path
- **Feature Engineering:** Created Cost Per Lifted User (CPLU) metric combining spend, reach, and absolute lift

### 3. Marketing Analytics & KPIs
- **Conversion Efficiency:** CPA analysis by market and channel
- **Brand Lift Modeling:** Calculated absolute lift percentages and volume of influenced users
- **Cost Per Lifted User (CPLU):** Primary optimization metric = `Spend / (Reach × Absolute_Lift)`
- **ROI Optimization:** Balanced brand building vs. direct response objectives

### 4. Data Visualization & Business Intelligence
- **Multi-dimensional Analysis:** Created scorecards comparing markets across conversion efficiency, cost efficiency, and brand impact
- **Channel Strategy Visualization:** Dual-axis charts showing cost vs. brand lift trade-offs
- **Creative Performance Reporting:** Comparative analysis of ad concepts by consideration score
- **Budget Allocation Modeling:** Pie charts and efficiency-weighted projections

### 5. Strategic Business Analysis
- **Market Segmentation:** Identified emerging vs. saturated markets based on efficiency metrics
- **Channel Selection:** Recommended dual-channel strategy (Search for conversions, YouTube/Social for brand shift)
- **Creative Optimization:** Selected "Life Hack" creative (5.51 consideration score) for brand channels; "Study With Me" and "Coding Help" for search
- **Budget Allocation:** 70% to high-efficiency markets (SA/EG), 30% to maintain presence in UK

---

## 💡 Key Business Recommendations

**Strategic Recommendation:**
- **Markets:** Focus 70% budget on SA/EG (emerging, high-efficiency); 30% on UK (maintain presence, shift preference)
- **Channels:** Search (40% budget) for conversions; YouTube (35%) + Social (25%) for brand building
- **Creative:** "Life Hack" for brand channels; "Study With Me"/"Coding Help" for search intent
- **Projected Outcome:** 1.3M student signups; 84% from high-efficiency SA/EG markets

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Programming** | Python (Pandas, Statsmodels, Matplotlib/Seaborn) |
| **Statistical Analysis** | Z-tests, Hypothesis Testing, Significance Testing |
| **Data Manipulation** | Excel (VLOOKUP, Pivot Tables), Pandas (merge, groupby) |
| **Visualization** | Python visualization libraries, PowerPoint |
| **Business Intelligence** | KPI Design, Marketing Analytics, ROI Modeling |

---

## 📂 Project Files

- **[Gemini_Campaign_Strategy.pptx](./Gemini_Campaign_Strategy.pptx)** — Complete strategic presentation
- **[Challenge Documentation](./Google_Step_Up_Challenge_How_to_guide.pdf)** — Project requirements and technical guide

---

## 🔗 Related Projects

- [Dissertation — Stock Price Volatility Prediction](../Dissertation_Stock_Prediction_Model) — Advanced ML modeling
- [Customer Conversion & Marketing Analytics](../Customer_Conversion_Prediction_Marketing) — Campaign optimization
- [Brand Analytics — Twitter Data](../Brand_Analytics_Twitter_Scraping) — Social sentiment analysis

---

**[← Back to Portfolio](../README.md)**
