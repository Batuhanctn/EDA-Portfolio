# Student Performance EDA and Regression Analysis

## Project Overview

This project involves a comprehensive Exploratory Data Analysis (EDA) and the application of regression models to the `Student_Performance.csv` dataset. The primary goal is to understand the key factors that influence student performance and to build a predictive model that can accurately estimate a student's Performance Index based on these factors.

---

## Dataset

The analysis is based on the `Student_Performance.csv` dataset, which contains the following features:

- **Hours Studied**: The number of hours the student studied.
- **Previous Scores**: The student's scores from previous exams.
- **Extracurricular Activities**: Whether the student participates in extracurricular activities (Yes/No).
- **Sleep Hours**: The average number of hours the student sleeps.
- **Sample Question Papers Practiced**: The number of sample question papers the student practiced with.
- **Performance Index**: The student's performance index, which is the target variable for our predictive models.

---

## Modeling and Evaluation

A `LinearRegression` model was trained to predict the `Performance Index`. The dataset was split into training (80%) and testing (20%) sets, and the features were scaled using `StandardScaler`.

### Linear Regression Results

The model demonstrated high predictive accuracy on the test set.

- **Mean Absolute Error (MAE)**: `1.63`
- **Mean Squared Error (MSE)**: `4.22`
- **Root Mean Squared Error (RMSE)**: `2.05`
- **R-squared (R²)**: `0.989`
- **Adjusted R-squared**: `0.989`

The high R-squared value indicates that the model explains approximately 98.9% of the variance in the Performance Index, suggesting a very strong fit.

### Model Coefficients

- **Intercept**: `55.22`
- **Coefficients**:
  - `Hours Studied`: 2.84
  - `Previous Scores`: 1.02
  - `Extracurricular Activities`: 0.62
  - `Sleep Hours`: 0.48
  - `Sample Question Papers Practiced`: 0.22

These coefficients show the expected increase in Performance Index for a one-unit increase in each feature, with `Hours Studied` having the most significant impact.

---

## Model Benchmarking with LazyPredict

To compare the performance of the `LinearRegression` model against other regression algorithms, the `LazyPredict` library was used. The results show that several models achieve similarly high performance, with minimal differences in their evaluation metrics.

### Top Performing Models (based on R² and RMSE)

| Model                     | Adjusted R-Squared | R-Squared | RMSE | Time Taken |
|---------------------------|--------------------|-----------|------|------------|
| `Lars`                    | 0.99               | 0.99      | 2.05 | 0.01       |
| `LassoLarsIC`             | 0.99               | 0.99      | 2.05 | 0.00       |
| `LinearRegression`        | 0.99               | 0.99      | 2.05 | 0.01       |
| `BayesianRidge`           | 0.99               | 0.99      | 2.05 | 0.00       |
| `RidgeCV`                 | 0.99               | 0.99      | 2.05 | 0.00       |

This comparison confirms that a linear model is highly effective for this dataset, as more complex models do not offer a significant improvement in predictive power.

---

## How to Run

1. Clone or download the repository files.
2. Install the required libraries: `pandas`, `numpy`, `scikit-learn`, and `lazypredict`.
3. Open the `8-Student_PerformanceEDA&Regression.ipynb` file in a Jupyter Notebook or JupyterLab environment and run the cells sequentially.
