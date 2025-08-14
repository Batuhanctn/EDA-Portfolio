# 💎 Diamonds Price Prediction Project

## 📌 Project Overview
The aim of this project is to **predict the price of diamonds** based on their characteristics using **Machine Learning algorithms**.  
We use the **Diamonds Dataset** which includes various features such as `carat`, `cut`, `color`, `clarity`, `depth`, and `table`.  

This project demonstrates:
- Data preprocessing
- Feature engineering
- Visualization
- Model training & evaluation
- Comparison of regression algorithms

---

## 📂 Dataset Information

The dataset is commonly used for regression tasks and contains **53,940 rows** and **10 features**.

| Feature   | Description |
|-----------|-------------|
| `carat`   | Weight of the diamond (0.2–5.01) |
| `cut`     | Quality of the cut (Fair, Good, Very Good, Premium, Ideal) |
| `color`   | Diamond color, from J (worst) to D (best) |
| `clarity` | Measurement of diamond clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF) |
| `depth`   | Total depth percentage = (z / mean(x, y)) * 100 |
| `table`   | Width of the top of the diamond relative to widest point (43–95) |
| `x`       | Length in mm (0–10.74) |
| `y`       | Width in mm (0–58.9) |
| `z`       | Depth in mm (0–31.8) |
| `price`   | Price in USD (326–18,823) |

---

## 🔧 Data Preprocessing

### 1. Handling Missing & Outlier Values
- Removed rows with invalid dimensions (`x`, `y`, `z` ≤ 0).
- Dropped extreme outliers in `depth` and `table`.

| Condition | Action |
|-----------|--------|
| `depth < 45` or `depth > 75` | Row dropped |
| `table < 40` or `table > 75` | Row dropped |
| `y > 20` | Row dropped |
| `z > 30` or `z < 2` | Row dropped |

---

### 2. Encoding Categorical Features
- `cut`, `color`, `clarity` converted using **One-Hot Encoding** and **Label Encoding** for testing performance differences.

| Encoding | Explanation | Example |
|----------|-------------|---------|
| **Label Encoding** | Converts categories to integer values. | `cut: Ideal → 0, Premium → 1 ...` |
| **One-Hot Encoding** | Creates binary columns for each category. | `cut_Ideal, cut_Premium, ...` |

---

### 3. Feature Engineering
- Added polynomial and interaction features:
  - `carat^2`, `depth * table`, `x*y*z`
- Normalized continuous variables using **StandardScaler**.

---

## 📊 Data Visualization

### Example: Price vs Carat
- Strong positive correlation between carat and price.

---

## 🤖 Models Used

| Model | R² Score | RMSE |
|-------|----------|------|
| **Linear Regression** | 0.88 | 1359 |
| **Ridge Regression** | 0.92 | 1105 |


---
