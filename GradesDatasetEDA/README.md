# Grades Dataset EDA & Linear Regression

## Project Overview

This project involves an Exploratory Data Analysis (EDA) of the Grades Dataset, followed by the development of a Linear Regression model to predict student exam scores. The primary goal is to analyze how factors like study hours, sleep patterns, class attendance, and social media usage correlate with academic performance and to build a predictive model based on these insights.

## Dataset Description

The dataset (`gradesdataset.csv`) contains information about student habits and their corresponding exam scores.

-   **Features**:
    -   `Study Hours`: The number of hours a student studies per week.
    -   `Sleep Hours`: The average number of hours a student sleeps per day.
    -   `Attendance Rate`: The student's class attendance percentage.
    -   `Social Media Hours`: The number of hours a student spends on social media per week.
-   **Target Variable**:
    -   `Exam Score`: The score achieved by the student in an exam.

The dataset consists of 50 samples with no missing values.

## Modeling and Evaluation

A Linear Regression model was trained to predict the `Exam Score` based on the four features.

### 1. Data Preprocessing
-   The data was split into training (75%) and testing (25%) sets.
-   Features were scaled using `StandardScaler`.

### 2. Model Performance

The model's performance was evaluated on the test set, yielding the following metrics:

| Metric             | Value   |
| ------------------ | ------- |
| MAE                | 3.277   |
| MSE                | 14.601  |
| RMSE               | 3.821   |
| R² Score           | 0.916   |
| Adjusted R² Score  | 0.874   |

### 3. Model Coefficients

The intercept and coefficients for the linear regression model are as follows:

-   **Intercept**: 77.41

| Feature              | Coefficient |
| -------------------- | ----------- |
| Study Hours          | 9.280       |
| Sleep Hours          | 1.790       |
| Attendance Rate      | 3.243       |
| Social Media Hours   | -4.081      |

## How to Run the Project

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Open the `GradesDataEDA&LinerRegression.ipynb` notebook in Jupyter.
4.  Run the cells sequentially to see the analysis and model results.
