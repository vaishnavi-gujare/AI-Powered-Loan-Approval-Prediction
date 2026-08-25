import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
csv_path = "loan_approval_dataset.csv"
if not os.path.exists(csv_path):
    # Try parent directory if run from models
    if os.path.exists(os.path.join("..", "loan_approval_dataset.csv")):
        csv_path = os.path.join("..", "loan_approval_dataset.csv")
    else:
        print("Error: loan_approval_dataset.csv not found in the project folder!")
        print("Please copy your downloaded 'loan_approval_dataset.csv' file into the 'loan' folder and run again.")
        exit(1)

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path, skipinitialspace=True)

# 2. Clean data
if 'loan_id' in df.columns:
    df = df.drop(columns=['loan_id'])
    print("Dropped loan_id column.")

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.strip()
print("Cleaned whitespaces.")

# Map target
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# 3. Features & Target
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# 4. Preprocessing Setup
numerical_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                      'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
categorical_features = ['education', 'self_employed']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# Fit preprocessor on full data
X_processed = preprocessor.fit_transform(X)
feature_names = (
    numerical_features + 
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
)

# 5. Train Random Forest Model locally
print("Training Random Forest Classifier locally to ensure version compatibility...")
model = RandomForestClassifier(random_state=42)
model.fit(X_processed, y)

# 6. Save files into models/ directory
os.makedirs("models", exist_ok=True)
joblib.dump(model, os.path.join("models", "loan_model.joblib"))
joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))
joblib.dump(feature_names, os.path.join("models", "feature_names.joblib"))

print("\n--- LOCAL TRAINING COMPLETE ---")
print("Saved: models/loan_model.joblib")
print("Saved: models/preprocessor.joblib")
print("Saved: models/feature_names.joblib")
print("These model files are now 100% compatible with your local VS Code environment!")
