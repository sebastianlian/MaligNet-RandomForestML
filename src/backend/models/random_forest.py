import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Handle imbalanced data
from colorama import Fore, Style, init  # Import colorama
import matplotlib.pyplot as plt
import seaborn as sns

init(autoreset=True)

# ✅ **Step 1: Load Dataset**
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "BRCA.csv")

df = pd.read_csv(DATA_PATH)

print(Fore.CYAN + Style.BRIGHT + "🔥 IN RANDOM_FOREST.PY 🔥")

# ✅ **Step 2: Rename Columns for Consistency**
df = df.rename(columns={
    "Age": "Age",
    "Protein1": "Protein1",
    "Protein2": "Protein2",
    "Protein3": "Protein3",
    "Protein4": "Protein4",
    "Tumour_Stage": "Stage",
    "Histology": "Histology",
    "Patient_Status": "RiskOfDeath",
    "Surgery_Type": "Surgery_type"
})

# ✅ **Step 3: Convert Categorical Columns to Numeric**
stage_mapping = {"I": 1, "II": 2, "III": 3}
df["Stage"] = df["Stage"].map(stage_mapping)
df["RiskOfDeath"] = df["RiskOfDeath"].map({"Alive": 0, "Dead": 1})

print(Fore.YELLOW + "✅ Unique values in RiskOfDeath:", df["RiskOfDeath"].unique())
# Factorize Histology & Surgery_Type
df["Histology"], _ = pd.factorize(df["Histology"])
df["Surgery_Type"], _ = pd.factorize(df["Surgery_type"])

# ✅ **Step 4: Handle Missing Values**
df = df.dropna(subset=["RiskOfDeath"])  # Ensure target column has no NaNs

# ✅ **Step 5: Feature Engineering**
df["Protein_Avg"] = df[["Protein1", "Protein2", "Protein3", "Protein4"]].mean(axis=1)  # Avg of protein levels
df["Histology_Stage"] = df["Histology"] * df["Stage"]  # Interaction feature
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 40, 60, 100], labels=[0, 1, 2])  # Binning Age

# **Step 6: Check Correlation**
correlation_matrix = df[["Histology", "Surgery_Type", "RiskOfDeath"]].corr()
print(Fore.BLUE + "Correlation Matrix:\n", correlation_matrix)
# **Step 7: Feature Selection**
features = ["Age", "Protein1", "Protein2", "Protein3", "Protein4", "Protein_Avg",
            "Stage", "Histology", "Surgery_Type", "Histology_Stage", "Age_Group"]

X = df[features]
y = df["RiskOfDeath"]

print(Fore.RED + "Missing values in RiskOfDeath:", df["RiskOfDeath"].isna().sum())
# **Step 8: Standardize Data**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 9: Apply SMOTE Before Splitting**
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# **Step 10: Split Data**
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# **Step 11: Hyperparameter Tuning (Grid Search)**
param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'max_features': ["sqrt"],
    'bootstrap': [True]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# **Step 12: Use Best Model Found**
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(Fore.GREEN + "Best Parameters Found by Grid Search:", best_params)
print(Fore.GREEN + "Best Cross-Validation Accuracy:", grid_search.best_score_)

# **Step 13: Train Best Model**
print(Fore.CYAN + "Training Best Model...")
best_model.fit(X_train, y_train)

# **Step 14: Make Predictions**
y_pred = best_model.predict(X_test)

# **Step 15: Evaluate Performance**
final_accuracy = accuracy_score(y_test, y_pred)

# **Step 16: Store Results**
rf_results = {
    "Accuracy": final_accuracy,
    "Cross-Validation Accuracy": grid_search.best_score_,
    "Feature Importance": dict(zip(features, best_model.feature_importances_))
}

# **Step 17: Print Results**
print(Fore.LIGHTYELLOW_EX + "🔥🔥🔥 Random Forest Analysis Complete: 🔥🔥🔥")
print(Fore.LIGHTGREEN_EX + "Final Model Accuracy:", final_accuracy)
print(Fore.LIGHTCYAN_EX + "Cross-Validation Accuracy:", grid_search.best_score_)

