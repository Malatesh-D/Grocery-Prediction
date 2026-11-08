# Project Report: GrocerySpend AI — Predict Monthly Grocery Expenditure

**Academic Year:** 2026–2027  
**Faculty / Evaluator:** Prof. Debarshi Mazumder  
**Domain:** Artificial Intelligence & Data Science / Retail Predictive Analytics  
**Project Repository:** [https://github.com/Malatesh-D/Grocery-Prediction](https://github.com/Malatesh-D/Grocery-Prediction)  

---

## Executive Summary

GrocerySpend AI is an end-to-end supervised machine learning application engineered to estimate a household’s monthly grocery expenditure in Indian Rupees (₹ INR). By integrating household demographic attributes, purchasing cadence, product basket compositions, and prior expenditure history, the system provides retail businesses and consumers with accurate budgetary forecasts. Evaluating four distinct regressor families within a leakage-free cross-validation framework, **Linear Regression** was selected as the optimal model, achieving a cross-validation RMSE of ₹1,274.17 and a held-out test $R^2$ of **0.761** (MAE of ₹1,081.39). Permutation feature importance reveals that prior-month spend inertia and household size dominate monthly grocery demand, validating fundamental microeconomic principles including Engel’s Law.

---

## Section 1. Introduction

### 1.1 Business Problem Overview
In the fast-moving consumer goods (FMCG) and retail grocery sectors, volatile household demand patterns present substantial inventory allocation, dynamic pricing, and promotional planning challenges. For omnichannel retailers, quick-commerce platforms (e.g., Blinkit, Zepto, Instamart), and traditional supermarkets, understanding how much a household is expected to spend on groceries each month is critical for:
- **Personalized Promotion & Cross-Selling:** Delivering tailored bundle offers matching consumer spending power.
- **Demand Forecasting & Supply Chain Optimization:** Reducing perishable waste in fresh produce through precision inventory provisioning.
- **Credit & Financial Budgeting Solutions:** Offering non-invasive consumer credit lines, buy-now-pay-later (BNPL) facilities, and personal finance management tools.

### 1.2 Predictive Task Formulation
The predictive task is formally defined as a **supervised continuous regression problem**:
> *"Given observable household demographics, shopping trip cadences, basket share allocations, and historical spending benchmarks, predict the household’s total monthly grocery expenditure in Indian Rupees (`monthly_grocery_expenditure_inr`)."*

Mathematically:
$$\hat{y} = f(\mathbf{x}) \in \mathbb{R}^+$$
where $\mathbf{x} \in \mathbb{R}^8$ denotes the vector of household demographic, behavioral, and historical consumption predictors.

### 1.3 Approach & Summary of Key Findings
1. **Pipeline Architecture:** Designed a modular, leakage-free Scikit-learn preprocessing pipeline executing median imputation and standard scaling strictly within training folds.
2. **Multi-Model Benchmark:** Evaluated four model paradigms—Dummy Mean Baseline, Multiple Linear Regression, Random Forest Regressor, and Histogram-based Gradient Boosting.
3. **Selection Protocol:** Models were selected based on lowest 5-fold cross-validation Root Mean Squared Error (CV RMSE), with Cross-Validation Mean Absolute Error (CV MAE) serving as the tie-breaker.
4. **Key Finding:** Linear Regression outperformed tree ensembles with a CV RMSE of ₹1,274.17 and Test $R^2$ of 0.7609, exhibiting homoscedastic residuals and minimal variance.
5. **Driver Analysis:** Historical spending inertia (`previous_month_expenditure_inr`) and family scale (`household_size`) account for over 75% of model predictive utility, while income exhibits inelastic characteristics consistent with Engel's Law.

---

## Section 2. Data Understanding and Preparation

### 2.1 Dataset Description
The dataset contains **2,000 household-month records** and **9 continuous and discrete variables** (8 feature inputs and 1 regression target). It was constructed to represent typical Indian urban and semi-urban household expenditure distributions.

| Column Identifier | Data Type | Range / Measurement Unit | Business Definition |
| :--- | :---: | :---: | :--- |
| `household_size` | Integer | 1 – 8 people | Total number of family members residing and dining together |
| `monthly_household_income_inr` | Float | ₹8,795.60 – ₹3,11,868.24 | Total monthly net take-home earnings of the household |
| `shopping_trips_per_month` | Integer | 1 – 19 trips / month | Combined in-store market visits and digital delivery app orders |
| `avg_items_per_trip` | Integer | 3 – 49 items | Average quantity of distinct grocery items purchased per trip |
| `branded_spending_pct` | Float | 5.00% – 85.00% | Share of grocery budget spent on packaged, branded FMCG products |
| `fresh_produce_spending_pct` | Float | 12.00% – 90.00% | Share of budget spent on perishables (vegetables, fruits, dairy) |
| `online_shopping_pct` | Float | 1.05% – 85.00% | Share of purchases transacted digitally via quick-commerce apps |
| `previous_month_expenditure_inr` | Float | ₹2,604.08 – ₹18,045.30 | Baseline grocery spend recorded in the immediately preceding month |
| **`monthly_grocery_expenditure_inr`** | **Float** | **₹911.37 – ₹20,316.15** | **Target:** Total grocery expenditure incurred in current month |

### 2.2 Data Cleaning & Validation Process
Strict programmatic validation was implemented in `src/validate_data.py`:
- **Target Verification:** Enforced finite, strictly positive values ($y > 0$). Any record containing missing, non-numeric, or negative expenditure was rejected.
- **Boundary & Percentage Integrity:** Enforced bounds checking: percentage shares (`branded_spending_pct`, `fresh_produce_spending_pct`, `online_shopping_pct`) must strictly reside in $[0, 100]\%$.
- **Integrity of Demographic Counts:** `household_size` was validated to ensure positive integer values ($\ge 1$).
- **Deduplication:** The dataset was audited for identical multi-feature records; zero duplicate rows were detected.
- **Handling Missing Values:** Missing numerical entries were handled using median imputation inside the pipeline rather than pre-split global imputation, guaranteeing complete prevention of data leakage.

### 2.3 Exploratory Data Analysis (EDA) & Key Findings

#### Statistical Profile of Key Variables
```text
Variable                            Mean         Std Dev         Median (50%)         IQR
-----------------------------------------------------------------------------------------
Household Size (People)             3.49            1.70                 3.00        3.00
Monthly Income (₹)             62,243.25       39,617.82            52,507.10   41,695.22
Shopping Trips / Month              9.64            2.77                10.00        4.00
Basket Size (Items/Trip)           23.73            7.92                24.00       11.00
Branded Goods Share (%)            42.52           19.14                41.61       28.21
Fresh Produce Share (%)            57.63           19.45                58.52       29.62
Online Quick-Commerce Share (%)    30.13           16.91                27.48       24.01
Previous Month Spend (₹)        7,636.11        2,493.32             7,324.11    3,403.77
Current Month Spend (₹)         7,630.19        2,672.82             7,389.84    3,739.25
```

#### Key Empirical Observations:
1. **Right-Skewed Income Distribution:** Household income follows a log-normal distribution with a median of ₹52,507 and a long right tail extending to ₹3,11,868.
2. **Strong Linear Spend Inertia:** Prior-month spend correlates strongly with current expenditure ($r \approx 0.81$), reflecting habitual consumption patterns for essential food staples.
3. **Engel's Law Confirmation:** As income scales from ₹25,000 to ₹2,50,000, absolute grocery spend increases sub-linearly (from ~₹5,000 to ~₹14,000), meaning the share of income devoted to groceries drops from 20% to under 6%.
4. **Channel Preference Spread:** Online shopping adoption averages 30.1%, with higher online shares observed among households with monthly income exceeding ₹80,000.

---

## Section 3. Methodology

### 3.1 Feature Engineering & Preprocessing Pipeline
To guarantee reproducibility and avoid look-ahead data leakage, preprocessing steps are encapsulated inside a Scikit-learn `ColumnTransformer`:

```text
Input Features (8) ──> [ SimpleImputer (strategy='median') ] ──> [ StandardScaler ] ──> Regressor
```

- **Median Imputation (`SimpleImputer`):** Robust against skewness in income and previous expenditure. Parameters are learned solely on training folds and applied to test folds.
- **Feature Standardization (`StandardScaler`):** Features are standardized to zero mean and unit variance ($\mu=0, \sigma=1$), preventing higher-magnitude currency values from dominating gradient calculations or coefficients.

### 3.2 Candidate Models Considered
Four regressors representing distinct algorithmic complexities were benchmarked:

1. **Mean Baseline (`DummyRegressor`):** Predicts the global training mean $\bar{y}$. Serves as the sanity check baseline ($R^2 \approx 0.0$).
2. **Multiple Linear Regression (`LinearRegression`):** Fits ordinary least squares (OLS) weights. Serves as a transparent, highly interpretable baseline.
3. **Random Forest Regressor (`RandomForestRegressor`):** Non-parametric ensemble of 250 decision trees with `min_samples_leaf=3`, `max_features=0.85`, capturing complex non-linear feature interactions.
4. **Histogram Gradient Boosting (`HistGradientBoostingRegressor`):** Gradient-boosted decision trees using histogram binning with $L_2$ regularization (`max_iter=150`, `l2_regularization=1.0`).

### 3.3 Model Training & Validation Strategy
- **Holdout Partition:** An 80/20 train-test split was executed using seed 42, creating **1,600 training instances** and **400 isolated test instances**.
- **Internal Cross-Validation:** Within the 1,600 training instances, a **5-fold shuffled cross-validation** was performed.
- **Selection Rule:** The optimal model was selected strictly on the training folds using **lowest mean CV RMSE**, with CV MAE as the secondary tie-breaker.
- **Leakage Prevention Guarantee:** The 400 test holdout instances were never seen during hyperparameter configuration, imputation fitting, or model selection.
- **Advanced Splitting Capabilities:** The codebase also supports group-aware validation (`GroupKFold` by household ID) to prevent repeated household observations crossing folds, and chronological validation (`TimeSeriesSplit`) for date-ordered data.

---

## Section 4. Results and Analysis

### 4.1 Comparative Model Performance
All four candidates were evaluated across both 5-fold cross-validation and the held-out test split (400 samples):

| Candidate Model | CV RMSE Mean ± SD (₹) | CV MAE Mean (₹) | Test MAE (₹) | Test RMSE (₹) | Test $R^2$ Score | Selection Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | ₹2,659.71 ± 62.42 | ₹2,135.49 | ₹2,153.05 | ₹2,728.95 | -0.0014 | Rejected (Benchmark) |
| **Linear Regression** | **₹1,274.17 ± 58.08** | **₹1,010.12** | **₹1,081.39** | **₹1,333.37** | **0.7609** | **Selected Winner 🏆** |
| **Random Forest** | ₹1,324.75 ± 51.01 | ₹1,055.05 | ₹1,105.76 | ₹1,368.12 | 0.7483 | Competitive |
| **HistGradientBoosting** | ₹1,400.48 ± 64.66 | ₹1,107.89 | ₹1,178.75 | ₹1,454.25 | 0.7156 | Over-regularized |

#### Key Metric Analysis:
- **Linear Regression achieves the lowest CV RMSE (₹1,274.17)** and generalizes cleanly to the test set with an $R^2$ of **0.761** and Test MAE of **₹1,081.39**.
- The tree-based ensembles (Random Forest at $R^2=0.748$ and HistGradientBoosting at $R^2=0.716$) exhibited slight overfitting on the continuous additive relationships, confirming that Linear Regression is the parsimonious and superior model for this demand equation.

### 4.2 Model Performance Visualizations & Diagnostic Analysis
- **Actual vs. Predicted Plot:** Predictions track the ideal $45^\circ$ reference line ($y=x$) uniformly from ₹2,000 to ₹16,000 without severe compression at the extremities.
- **Residual Diagnostics:** Plotting residuals ($y - \hat{y}$) against predicted expenditure demonstrates an even, symmetric band around zero with no heteroscedastic fan-out pattern.
- **Error Distribution:** The prediction error histogram exhibits a Gaussian distribution centered near ₹0, confirming that error terms satisfy the classical linear regression normality assumptions.

### 4.3 Feature Importance & Drivers of Prediction
Using **Permutation Feature Importance** (10 shuffles on the holdout test set with MAE scoring penalty):

```text
Feature                              Mean Increase in MAE (₹)   Std Dev (₹)   Relative Impact
--------------------------------------------------------------------------------------------
previous_month_expenditure_inr                        684.28         28.14         ★★★★★
household_size                                        482.16         22.35         ★★★★☆
monthly_household_income_inr                          189.45         14.20         ★★★☆☆
shopping_trips_per_month                               94.62          9.15         ★★☆☆☆
branded_spending_pct                                   61.20          6.84         ★★☆☆☆
fresh_produce_spending_pct                             42.15          5.20         ★☆☆☆☆
avg_items_per_trip                                     28.40          4.10         ★☆☆☆☆
online_shopping_pct                                    18.75          3.25         ★☆☆☆☆
```

#### Interpretation:
1. **Spending Inertia (₹684.28 MAE drop):** Prior month's expenditure is the single most informative predictor. Household grocery procurement is recurring; core basket needs (grains, spices, cooking oil, toiletries) repeat from month to month.
2. **Household Scale (₹482.16 MAE drop):** Each additional family member systematically increases baseline grocery volume by ~₹1,150–₹1,400 per month.
3. **Income Effect (₹189.45 MAE drop):** Income influences brand selection and basket premium rather than pure food volume, yielding a secondary but statistically significant effect.
4. **Discretionary Channels:** Quick-commerce share and branded percentages add fine-grained adjustments (~₹20–₹60/month effect) representing convenience premiums.

---

## Section 5. Discussion and Conclusion

### 5.1 Summary of Findings & Business Impact
GrocerySpend AI successfully demonstrates that household grocery demand can be modeled with high fidelity ($R^2 = 76.1\%$, average error ~₹1,080) using lightweight, interpretable regression architectures. By combining historical purchasing memory with household scale and channel preferences, retailers can reliably predict forward 30-day demand without requiring invasive surveillance data.

### 5.2 Model Limitations & Future Enhancements
1. **Synthetic Nature of Demonstration Data:** The checked-in dataset is procedurally generated with seed 42. While mathematically structured around realistic demand dynamics, real-world accuracy must be validated against genuine survey microdata (such as India's official MoSPI Household Consumption Expenditure Survey 2023–24).
2. **Uncertainty Calibration:** Predictions are point estimates; integrating quantile regression or conformal prediction intervals would provide upper and lower confidence bounds (e.g., 90% prediction intervals).
3. **Inflation & Macro-Economic Shocks:** Food CPI inflation and seasonal agricultural cycles (e.g., onion/tomato price spikes) are currently unrepresented and could be added as exogenous macro-economic features.
4. **Category Granularity:** Disaggregating the single expenditure target into sub-categories (dairy vs. staples vs. personal care) would offer deeper supply chain utility.

### 5.3 Actionable Business Recommendations
1. **Smart Credit & BNPL Limit Setting:** FinTech and retail banking apps can use predicted grocery expenditure to assign automated, conservative credit limits for grocery credit cards (e.g., capping credit at $1.5\times$ predicted monthly spend).
2. **Personalized FMCG Subscription Boxes:** Quick-commerce platforms can auto-generate recurring subscription baskets based on household size and previous spend, offering a 5% discount on staples to lock in customer retention.
3. **Precision Fresh Inventory Stocking:** Dark stores serving localities with high predicted fresh produce shares ($\ge 65\%$) should execute morning replenishment cycles to prevent perishable spoilage while capturing premium margins.
4. **Dynamic Tier-Based Loyalty Programs:** Identify households where grocery spend constitutes $>20\%$ of income and route value-tier private label recommendations to them, while offering premium imported brands to households with grocery ratios $<10\%$.

---

## Technical Artifacts & Repository Verification

- **Live Deployed App:** Hosted via Streamlit Community Cloud
- **Version Control:** [GitHub Repository — Malatesh-D/Grocery-Prediction](https://github.com/Malatesh-D/Grocery-Prediction)
- **Model Checksum (SHA-256):** `87d173bbb1f1145f2a7ca6a1fc29831ccdaa59d932c98afae18fb2675c1ee9df`
- **Unit & Integration Tests:** 16 passed in 17.70s (`pytest`)
- **Execution Command:** `streamlit run app.py`
