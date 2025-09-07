### Income Prediction with Random Forest Classifier
*Income Classification Project: Random Forest Application*

### Overview
This project aims to develop a classification model using a dataset of individuals' characteristics in the USA to predict whether their annual income is above or below $50,000. The project includes a detailed Exploratory Data Analysis (EDA) to understand the dataset's structure and a machine learning model built using a Random Forest Classifier to find the best-performing model.

### Table of Contents
* [Dataset](#dataset)
* [Libraries Used](#libraries-used)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
* [Model Building and Evaluation](#model-building-and-evaluation)
* [Setup](#setup)

---

### Dataset
The dataset used in this project is `income_evaluation.csv`. It contains a total of **32,561 observations** and **15 variables**. The variables include demographic information such as age, occupation, education level, marital status, and hours worked per week.

**All Variables in the Dataset:**
* `age`: Age (Numerical)
* `workclass`: Work Class (Categorical)
* `finalweight` (original name: `fnlwgt`): A final weight used for demographic estimation. (Numerical)
* `education`: Education Level (Categorical)
* `education_num` (original name: `education-num`): Numerical equivalent of the education level. (Numerical)
* `marital_status`: Marital Status (Categorical)
* `occupation`: Occupation (Categorical)
* `relationship`: Family Relationship (Categorical)
* `race`: Race (Categorical)
* `sex`: Sex (Categorical)
* `capital_gain`: Capital Gain (Numerical)
* `capital_loss`: Capital Loss (Numerical)
* `hours_per_week`: Hours Worked Per Week (Numerical)
* `native_country`: Native Country (Categorical)
* `income`: Target variable, whether income is `<=50K` or `>50K`. (Categorical)

---

### Libraries Used
* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **seaborn & matplotlib:** For data visualization.
* **scikit-learn:** For machine learning model building, data preprocessing, and model evaluation.

---

### Exploratory Data Analysis (EDA)
The EDA phase involved a deep dive into the dataset to understand its structure. Key analyses performed include:

* **Data Types and Missing Values:** The dataset has 6 numerical and 9 categorical columns. There are no nominally missing values. However, some categorical variables like `workclass`, `occupation`, and `native_country` contain ' ?' as a marker for missing data. These values were later filled with the mode.

* **Univariate Analysis:** The distribution of each variable was examined, and the unique values and counts for categorical variables were checked.

---

### Data Preprocessing and Feature Engineering
To improve the model's performance, the following data preprocessing and feature engineering steps were performed:

1.  **Column Renaming:** Some original column names were renamed to improve readability and remove spaces (e.g., `fnlwgt` -> `finalweight`, `education-num` -> `education_num`, `capital-gain` -> `capital_gain`).
2.  **Handling Missing Values:** The ' ?' missing values in the categorical columns (`workclass`, `occupation`, and `native_country`) were imputed with the mode of each respective column. This prevents data loss and helps improve model performance.
3.  **Variable Separation:** The dataset was split into numerical and categorical columns to apply different preprocessing steps tailored to each data type.
4.  **Target Variable Encoding:** The target variable `income` was converted from categorical to numerical values using `LabelEncoder`. This transformed `<=50K` to 0 and `>50K` to 1, making it a suitable format for the model.
5.  **Data Scaling:** To mitigate the effect of outliers and ensure that variables with different scales are weighted equally by the model, numerical variables were scaled using **RobustScaler**.
6.  **Training and Test Split:** The dataset was split into a 70% training set and a 30% test set to objectively evaluate the model's performance.

---

### Model Building and Evaluation
A **Random Forest Classification Model** was used in this project. Hyperparameter tuning was performed with `RandomizedSearchCV` to maximize the model's performance.

**Best Model Hyperparameters:**
* `n_estimators`: 200
* `min_samples_split`: 2
* `max_features`: 'sqrt'
* `max_depth`: 15

**Model Performance Metrics:**
The trained model produced the following results on the test dataset:
* **Accuracy Score:** `0.8529`
* **Classification Report:**
    * For class `0` (`<=50K`), `precision` was `0.88`, `recall` was `0.94`, and `f1-score` was `0.91`.
    * For class `1` (`>50K`), `precision` was `0.78`, `recall` was `0.60`, and `f1-score` was `0.68`.
* **Confusion Matrix:**
    ```
    [[6996  411]
     [ 945 1417]]
    ```
    This matrix indicates that the model correctly predicted 6996 observations as `<=50K` and 1417 observations as `>50K`.

### Setup
To run this project locally, you can install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
