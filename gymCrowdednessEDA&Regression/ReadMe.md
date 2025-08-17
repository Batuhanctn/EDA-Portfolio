### 🏋️ Gym Crowdedness Prediction with Regression Models 📈
*A Regression Project for Predicting Gym Occupancy*

### Overview
This project aims to predict the number of people (crowdedness) in a gym at different times using various regression models. The process includes a detailed Exploratory Data Analysis (EDA) to understand the dataset, followed by a comparison of different regression models to determine the most suitable one for the task. Specifically, **Decision Tree Regression** and **KNN Regression** models were fine-tuned with hyperparameter optimization.

---

### 📊 Table of Contents
* [Dataset](#dataset)
* [Libraries Used](#libraries-used)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Data Preprocessing](#data-preprocessing)
* [Model Building and Evaluation](#model-building-and-evaluation)
* [Conclusion](#conclusion)
* [Setup](#setup)

---

### 📁 Dataset
The dataset used for this project is `15-gym_crowdedness.csv`. It contains a total of **62,184 observations** and **11 variables**.

**All Variables in the Dataset:**
* `number_people`: The number of people in the gym (Target variable).
* `date`: Date and time.
* `timestamp`: Unix timestamp.
* `day_of_week`: Day of the week (0: Sunday - 6: Saturday).
* `is_weekend`: Whether it's a weekend (1: Yes, 0: No).
* `is_holiday`: Whether it's a public holiday (1: Yes, 0: No).
* `temperature`: Outside temperature.
* `is_start_of_semester`: Whether it's the start of the semester (1: Yes, 0: No).
* `is_during_semester`: Whether it's during the semester (1: Yes, 0: No).
* `month`: Month.
* `hour`: Hour.

---

### 💻 Libraries Used
* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **seaborn & matplotlib:** For data visualization.
* **scikit-learn:** For building and evaluating machine learning models.

---

### 🔍 Exploratory Data Analysis (EDA)
The EDA phase provided a deep understanding of the dataset's structure. Key analyses performed include:
* **Data Type and Shape:** The dataset has a shape of `(62184, 11)`, and the `date` variable was identified as an `object` type.
* **Missing Values:** There are no nominally missing values (`null`) in the dataset.
* **Statistical Summary:** The statistical summary of the dataset was reviewed to understand the distribution and central tendencies of each column.

---

### 🧹 Data Preprocessing
To ensure the models perform optimally, the following data preprocessing steps were performed:
1.  The `date` column was converted to a `datetime` format.
2.  A new `year` feature was extracted from the `date` column.
3.  The original `date` column was dropped from the dataset.
4.  Categorical variables (`day_of_week`, `is_weekend`, `is_holiday`, `is_start_of_semester`, `is_during_semester`, `month`, `hour`, `year`) were converted to numerical features using `One-Hot Encoding`.
5.  The data was split into training (70%) and testing (30%) sets to objectively evaluate model performance.

---

### 🤖 Model Building and Evaluation
Several regression models were used and their performances were compared.

**Models Used:**
* Linear Regression
* Lasso
* Ridge
* K-Neighbors Regressor
* Decision Tree Regression
* Random Forest Regression

**Performance Metrics:**
The following metrics were used to evaluate the models' performance:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Squared Error (MSE)
* R-squared Score (Skor)

**Model Performance Results (Test Set):**

| Model Name | MAE | RMSE | MSE | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 10.78 | 14.45 | 208.82 | 0.599 |
| Lasso | 11.22 | 14.97 | 224.21 | 0.569 |
| Ridge | 10.78 | 14.45 | 208.82 | 0.599 |
| K-Neighbors Regressor | 5.05 | 7.53 | 56.75 | 0.891 |
| Decision Tree Regressor | 4.35 | 6.56 | 42.98 | 0.917 |
| Random Forest Regressor | 4.30 | 6.44 | 41.47 | 0.920 |

---

### ⚙️ Hyperparameter Tuning
To achieve the best possible performance, hyperparameter tuning was performed on the K-Neighbors and Random Forest Regressor models using **`RandomizedSearchCV`**.

**Hyperparameter Tuned Model Results (Test Set):**

| Model Name | Tuned Parameters | MAE | RMSE | MSE | R² Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K-Neighbors Regressor | `n_neighbors=2` | 4.64 | 6.95 | 48.27 | 0.907 |
| Random Forest Regressor | `n_estimators=500`, `max_features=7`, `max_depth=None`, `min_samples_split=2` | 4.29 | 6.42 | 41.21 | 0.921 |

### 🚀 Conclusion
Based on the comparison of both baseline and fine-tuned models, the **Random Forest Regressor** model achieved the highest R² score and the lowest MAE and RMSE values. This suggests that the complex structure of the dataset is more effectively captured by an ensemble learning model like Random Forest. The project's findings demonstrate that external factors (temperature, time, and day) are highly effective in predicting gym crowdedness.

---

### ➡️ Setup
To run this project locally, you can install the necessary libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
