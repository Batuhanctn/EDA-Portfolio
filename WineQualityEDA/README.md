# Wine Quality Exploratory Data Analysis (EDA)

## Project Overview

This project performs an Exploratory Data Analysis (EDA) on the Wine Quality dataset (`WineQT.csv`). The analysis focuses on understanding the relationships between various chemical properties of red wine and its perceived quality score. The notebook investigates feature distributions, correlations, and other insights without building a predictive model.

## Dataset Description

The dataset contains the following chemical properties of red wine, along with a quality rating:

-   **fixed acidity**: The amount of non-volatile acids in the wine.
-   **volatile acidity**: The amount of acetic acid in the wine, which can lead to an unpleasant vinegar taste at high levels.
-   **citric acid**: Can add 'freshness' and flavor to wines.
-   **residual sugar**: The amount of sugar remaining after fermentation stops.
-   **chlorides**: The amount of salt in the wine.
-   **free sulfur dioxide**: The free form of SO2; it prevents microbial growth and wine oxidation.
-   **total sulfur dioxide**: The amount of free and bound forms of S02.
-   **density**: The density of the wine.
-   **pH**: Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic).
-   **sulphates**: A wine additive that can contribute to sulfur dioxide gas levels, which acts as an antimicrobial and antioxidant.
-   **alcohol**: The percentage of alcohol content in the wine.
-   **quality**: A quality score between 0 and 10 (the target variable).
-   **Id**: A unique identifier for each sample.

## Analysis Goal

The primary objective of this EDA is to explore the dataset to identify which chemical properties most significantly influence the quality of red wine. The analysis relies on statistical summaries and data visualization to uncover key relationships and patterns.

## How to Run the Project

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```
3.  Open the `13-WineQualityEDA.ipynb` notebook in Jupyter.
4.  Run the cells sequentially to view the exploratory analysis.
