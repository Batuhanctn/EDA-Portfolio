# Exploratory Data Analysis (EDA) Portfolio

This repository contains a collection of my Exploratory Data Analysis (EDA) projects. Each project explores a different dataset to uncover insights, visualize data, and, in some cases, apply basic machine learning models.

## Projects

Here are the projects included in this portfolio:

1.  **[Student Performance EDA & Regression](./StudentPerformanceEDA&RegressionwithLazyPredict/)**
    -   An analysis of student performance data, including the development of a Linear Regression model to predict performance index based on study habits and other factors. This project also features a `LazyPredict` implementation to benchmark multiple regression models.

2.  **[Grades Dataset EDA & Linear Regression](./GradesDatasetEDA/)**
    -   Explores a dataset of student grades and builds a Linear Regression model to predict exam scores based on study habits, attendance, and social media usage.

3.  **[Wine Quality EDA](./WineQualityEDA/)**
    -   A detailed exploratory analysis of the chemical properties of red wine to determine their influence on its quality score. This project focuses purely on data visualization and statistical analysis without predictive modeling.

4.  **[Diamonds Dataset EDA](./DiamondsEDA&Regression/)**
    -   An exploratory data analysis of the Diamonds dataset, focusing on the relationship between diamond features (such as carat, cut, color, and clarity) and price. The project includes visualizations to highlight trends, correlations, and distributions within the dataset. Special emphasis is placed on understanding how different categorical and numerical attributes affect diamond pricing.

5.  **🏋️ [Gym Crowdedness EDA & Regression](./gymCrowdednessEDA&Regression/)**
    -   An analysis of a gym crowdedness dataset, exploring factors like temperature, time of day, and day of the week to build a regression model that predicts the number of people. It features a comparison of multiple models, including **Linear Regression**, **Decision Tree**, and **K-Neighbors Regressor**, along with hyperparameter tuning.

6.  **📈 [Income Prediction with Random Forest Classifier](./IncomeEvaluationwithRandomForestClassification/)**
    -   This project focuses on a comprehensive EDA and a machine learning application to predict whether a person's income is above or below $50,000. It includes detailed data preprocessing and feature engineering steps, culminating in the development and fine-tuning of a **Random Forest Classifier** model.
7.  **🚗 [Car Price Regression with Adaboost Regression](./CarPriceRegressionwithAdaboostRegression/)**
    -   An in-depth analysis of a second-hand car dataset to predict vehicle prices. This project features extensive data preprocessing and a mixed-strategy encoding (**Frequency Encoding** for high-cardinality features and **One-Hot Encoding** for others). The core of the project is the development and optimization of an **AdaBoost Regressor** model using **RandomizedSearchCV** to fine-tune hyperparameters.

8.  **🌍 [Country Development Analysis: Unsupervised Learning](./CountryUnsupervised/)**
    -   A comprehensive unsupervised learning project that analyzes socioeconomic data from 167 countries to classify them based on budget allocation needs. The project features **PCA** for dimensionality reduction and compares multiple clustering algorithms (**K-Means**, **Hierarchical Clustering**, **DBSCAN**, **HDBSCAN**) to categorize countries into "Budget Needed", "In Between", and "No Budget Needed" groups. Includes interactive geographical visualizations using **Plotly**.

Each project folder contains a Jupyter Notebook (`.ipynb`) with the complete analysis and the corresponding dataset (`.csv`). For more detailed information, please refer to the `README.md` file within each project's directory.


## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Plotly
-   Scikit-learn
-   HDBSCAN
-   LazyPredict
-   Jupyter Notebook
