# Used Car Price Prediction

A machine learning project for predicting used car prices based on vehicle specifications and other relevant features.

## Project Overview

The goal of this project is to build regression models that can predict the price of used cars using features such as vehicle age, mileage, engine size, power, location, fuel type, transmission, and brand.

Several machine learning models were trained and evaluated, and hyperparameter tuning was applied to selected models using `GridSearchCV`.

## Dataset

The dataset contains information about used cars, including:

* Car name and brand
* Location
* Manufacturing year
* Kilometers driven
* Fuel type
* Transmission type
* Owner type
* Mileage
* Engine
* Power
* Number of seats
* Car price

## Data Preprocessing

The following preprocessing steps were applied:

* Missing numerical values were handled using median imputation.
* Missing categorical values were handled using the most frequent value.
* Numerical features were standardized using `StandardScaler`.
* Categorical features were converted using `OneHotEncoder`.
* `ColumnTransformer` and `Pipeline` were used to keep preprocessing and modeling together.

Additional feature engineering was performed by creating:

* `Car_Age`

## Models

The following regression models were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest
5. Gradient Boosting

For Ridge, Lasso, Random Forest, and Gradient Boosting, hyperparameter tuning was performed using `GridSearchCV`.

## Evaluation Metrics

The models were evaluated using:

* **R²** — measures how much of the variance in the target variable is explained by the model.
* **MAE (Mean Absolute Error)** — measures the average absolute prediction error.
* **RMSE (Root Mean Squared Error)** — measures prediction error while giving larger errors more weight.

## Model Comparison

| Model             | Train R² | Validation R² |        MAE |       RMSE |
| ----------------- | -------: | ------------: | ---------: | ---------: |
| Gradient Boosting |   0.9686 |    **0.9323** | **1.5209** | **3.1366** |
| Random Forest     |   0.9825 |        0.9082 |     1.5614 |     3.6531 |
| Linear Regression |   0.7780 |        0.5977 |     3.2774 |     7.6477 |
| Lasso Regression  |   0.7779 |        0.5967 |     3.2815 |     7.6571 |
| Ridge Regression  |   0.7777 |        0.5938 |     3.2937 |     7.6847 |

## Hyperparameter Tuning

For Gradient Boosting, `GridSearchCV` was used with the following parameters:

* `n_estimators`: 100, 200
* `learning_rate`: 0.05, 0.1
* `max_depth`: 3, 5

The best configuration was:

```text
n_estimators = 200
learning_rate = 0.1
max_depth = 3
```

The best cross-validation R² score was approximately:

```text
CV R² = 0.87195
```

## Final Model

Based on the validation results, **Gradient Boosting** achieved the highest validation R² score among the tested models.

The tuned Gradient Boosting model was therefore selected as the final model and retrained using the complete training dataset before generating predictions for the test dataset.

## Prediction Output

The final model generated predictions for 1,505 test observations.

The predictions were saved in:

```text
submission.csv
```

## Project Structure

```text
used-car-price-prediction/
│
├── used_cars.ipynb
├── submission.csv
├── .gitignore
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Git & GitHub

## Key Takeaways

This project demonstrates a complete machine learning regression workflow, including:

* Data preprocessing
* Feature engineering
* Categorical encoding
* Numerical scaling
* Regression modeling
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Final model selection
* Generating predictions for unseen data

The results show that tree-based ensemble methods, particularly Gradient Boosting, were substantially more effective for this dataset than the linear regression-based models.
