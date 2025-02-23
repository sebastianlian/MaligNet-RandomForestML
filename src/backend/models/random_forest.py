import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "BRCA.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("IN RANDOM_FOREST.PY")
# Rename only the necessary columns
df = df.rename(columns={
    "Age": "Age",
    "Protein1": "Protein1",
    "Protein2": "Protein2",
    "Protein3": "Protein3",
    "Protein4": "Protein4",
    "Tumour_Stage": "Stage",
    "Histology": "Histology",
    "Patient_Status": "RiskOfDeath"
})

# Convert categorical columns to numeric
stage_mapping = {"I": 1, "II": 2, "III": 3}
df["Stage"] = df["Stage"].map(stage_mapping)
df["RiskOfDeath"] = df["RiskOfDeath"].map({"Alive": 0, "Dead": 1})

print("Unique values in RiskOfDeath:", df["RiskOfDeath"].unique())

df["Histology"], histology_labels = pd.factorize(df["Histology"])

# Remove rows where RiskOfDeath is NaN BEFORE selecting features
df = df.dropna(subset=["RiskOfDeath"])

# Select relevant features
features = ["Age", "Protein2", "Protein4", "Stage"]  # Using best features
X = df[features]
y = df["RiskOfDeath"]

print("Missing values in RiskOfDeath:", df["RiskOfDeath"].isna().sum())
df = df.dropna(subset=["RiskOfDeath"])  # Remove rows where RiskOfDeath is NaN

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

# Store results
rf_results = {
    "Accuracy": accuracy,
    "Feature Importance": dict(zip(features, model.feature_importances_))
}

# Print results for debugging
print("Random Forest Analysis Complete:", rf_results)
