import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "BRCA.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("IN LINEAR_REGRESSION.PY")

# Rename only the necessary columns
df = df.rename(columns={
    "Age": "Age",
    "Protein1": "Protein1",
    "Protein2": "Protein2",
    "Protein3": "Protein3",
    "Protein4": "Protein4",
    "Tumour_Stage": "Stage",
    "Histology": "Histology",
    "Patient_Status": "RiskOfDeath"  # Assuming Patient_Status represents survival risk
})

# Convert Stage from Roman numerals to numbers
stage_mapping = {"I": 1, "II": 2, "III": 3}
df["Stage"] = df["Stage"].map(stage_mapping)

# Manually map Histology categories to numbers
histology_mapping = {
    "Infiltrating Ductal Carcinoma": 0,
    "Mucinous Carcinoma": 1,
    "Infiltrating Lobular Carcinoma": 2
}

df["Histology"] = df["Histology"].map(histology_mapping)

# Convert RiskOfDeath from text to numbers
df["RiskOfDeath"] = df["RiskOfDeath"].map({"Alive": 0, "Dead": 1})

# Select relevant columns
columns_needed = ["Age", "Protein1", "Protein2", "Protein3",
                  "Protein4", "Stage", "Histology", "RiskOfDeath"]
df = df[columns_needed].dropna()  # Remove missing values

# Define the independent variables (X) and dependent variable (y)
features = [
    # x and ys
    (["Age", "Protein1", "Protein2", "Protein3", "Protein4"], "RiskOfDeath"),  # Age + All Proteins
    (["Age", "Stage"], "RiskOfDeath"),  # Age + Stage
    (["Age", "Histology"], "RiskOfDeath")  # Age + Histology
]

results = {}

for feature_set, target in features:
    X = df[feature_set]  # Independent variables
    y = df[target]  # Dependent variable (Risk of Death)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict values
    y_pred = model.predict(X_test)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    # This is saying that the model are not fitting the data well. The features may not be strong predictors of RiskOfDeath, or there is a data scaling issue
    results[f"{' + '.join(feature_set)} → RiskOfDeath"] = {
        # shows the relationship between each feature and RiskOfDeath
        "Coefficients": model.coef_.tolist(),

        # The predicted value of the RiskOfDeath when all features are 0
        "Intercept": model.intercept_,

        # How far prediction are from actual values (the lower the better)
        "Mean Squared Error": mse,

        #  1.0 -> perfect correlation, 0.0 -> no correlation, negative values -> performs worse than random choice
        "R-squared Score": r2
    }

# Print results for debugging
# print("Regression Analysis Complete:", results)

# print(df.shape)

print(df.corr()["RiskOfDeath"])
# Protein 2 and Protein 4 have the highest correlation with RiskOfDeath (0.083, 0.079)
# Stage is weakly correlated (0.0529) wit RiskOfDeath, which suggest it might not be a strong preditor
# Age has almost no correlation (0.0109), meaning it does not contribute significantly to predicting RiskOfDeath
# Histology has a negative weak correlation (-0.0140), meaning it's not a strong predictor
# All correlations are low (close to 0), this means that the current features are not a strong prediction of RiskOfDeath