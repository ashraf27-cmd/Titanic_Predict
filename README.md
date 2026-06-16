# Titanic Survival Prediction

A Machine Learning Classification project using the Titanic dataset. This project focuses on data cleaning, exploratory data analysis (EDA), feature engineering, and survival prediction using multiple classification algorithms.

---

## Project Overview

The objective of this project is to predict passenger survival on the Titanic using demographic and travel-related information.

The project includes:

* Data Cleaning & Preprocessing
* Missing Value Handling
* Exploratory Data Analysis (EDA)
* Feature Encoding
* Correlation Analysis
* Logistic Regression
* Decision Tree Classification
* Random Forest Classification
* Model Performance Comparison

---

## Dataset Information

The dataset contains passenger information from the Titanic disaster.

### Features

* PassengerId
* Survived
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

### Data Cleaning

The following preprocessing steps were performed:

* Removed duplicate records
* Dropped the Cabin column due to excessive missing values
* Filled missing Age values using the median age
* Filled missing Embarked values using the mode
* Encoded categorical variables (Sex and Embarked)
* Removed non-informative columns such as PassengerId, Name, and Ticket

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Exploratory Data Analysis

Several visualizations were created to understand passenger characteristics and survival patterns.

### Visualizations

* Survival Distribution
* Survival by Gender
* Age Distribution
* Survival by Passenger Class
* Correlation Heatmap

### Key Findings

* Female passengers had significantly higher survival rates.
* First-class passengers had a greater chance of survival.
* Passenger fare showed a positive relationship with survival.
* Passenger class and gender were among the strongest predictors.

---

## Machine Learning Models

### Logistic Regression

A linear classification model used as a baseline for survival prediction.

### Decision Tree Classifier

A tree-based classification model capable of capturing non-linear relationships.

### Random Forest Classifier

An ensemble learning model combining multiple decision trees to improve predictive performance.

---

## Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Model Comparison

| Model               | Accuracy                   |
| ------------------- | -------------------------- |
| Logistic Regression | Generated during execution |
| Decision Tree       | Generated during execution |
| Random Forest       | Generated during execution |

---

## Repository Structure

```text
Titanic_Predict/
│
├── Titanic.csv
├── Titanic.py
├── README.md
├── LICENSE
├── .gitignore
│
└── Viz/
    ├── survival_dist.png
    ├── gender_vs_survival.png
    ├── age_distribution.png
    ├── survival_by_passenger.png
    └── correlation_map.png
```

---

## Conclusion

This project demonstrates a complete machine learning workflow, from data preprocessing and exploratory analysis to model training and evaluation. Multiple classification algorithms were implemented and compared to identify the most effective approach for predicting passenger survival.
The analysis highlights the importance of passenger class, gender, and fare in determining survival outcomes and showcases practical applications of supervised machine learning techniques.

---
