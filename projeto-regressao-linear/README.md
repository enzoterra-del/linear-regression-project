# 📈 Linear Regression: Predicting Grades from Study Hours

Beginner Machine Learning project in Python. The model learns the relationship between **study hours** and **exam grades** using Linear Regression.

## What the Project Uses

* **pandas**: load and explore the data
* **scikit-learn**: `train_test_split`, `LinearRegression`, and evaluation metrics
* **matplotlib**: visualize the results

## Project Structure

```text
├── data/study_grades.csv   # dataset (200 rows)
├── generate_data.py        # generates the fictional dataset
├── main.py                 # trains, evaluates, and plots the model
├── requirements.txt
└── result.png              # generated charts
```

## How to Run

```bash
pip install -r requirements.txt
python generate_data.py   # optional, the CSV is already included
python main.py
```

## How It Works

1. Loads the CSV file using pandas
2. Separates `X` (study hours) and `y` (grade)
3. Splits the data into 80% training and 20% testing using `train_test_split`
4. Trains the `LinearRegression` model on the training data
5. Evaluates the model on test data that the model has never seen
6. Plots the regression line and the actual vs. predicted results

## Results

| Metric | Value       |
| ------ | ----------- |
| R²     | ~92%        |
| MAE    | ~4.2 points |
| RMSE   | ~5.2 points |

Learned equation:

```text
grade ≈ 30 + 6.5 × study_hours
```

![Result](result.png)

## About "Accuracy" in Regression

Accuracy (%) is mainly used for **classification** problems.

In regression, the model predicts continuous numerical values, so we use metrics such as:

* **R²**: measures how much of the variation in the grades the model explains. Closer to 100% generally means the model explains more of the variation.
* **MAE**: the average prediction error, measured in grade points.
* **RMSE**: similar to MAE, but gives more weight to larger errors.

## What I Learned

Through this project, I learned the fundamentals of **Linear Regression** and how a Machine Learning model can learn relationships between numerical variables.

I learned how to:

* Load and explore datasets using **pandas**
* Separate input features (`X`) from the target (`y`)
* Split data into training and testing sets
* Train a Linear Regression model using **scikit-learn**
* Make predictions with a trained model
* Evaluate regression models using **R², MAE, and RMSE**
* Visualize predictions and regression lines using **matplotlib**
* Understand the difference between **classification and regression**
* Understand why accuracy is not the main metric for regression problems
* Build a complete Machine Learning project from data preparation to evaluation
