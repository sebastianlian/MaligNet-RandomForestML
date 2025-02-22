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
    results[f"{' + '.join(feature_set)} → RiskOfDeath"] = {
        "Coefficients": model.coef_.tolist(),
        "Intercept": model.intercept_,
        "Mean Squared Error": mse,
        "R-squared Score": r2
    }

# Print results for debugging
print("Regression Analysis Complete:", results)

print(df.shape)

print(df.corr()["RiskOfDeath"])
