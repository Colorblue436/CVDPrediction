Cell 1: Imports and Setup
Python
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# Load your dataset
# df = pd.read_csv("./cardio.csv") 
# Assuming 'df' is already loaded from your existing cvd_pipeline logic
Cell 2: Training Function (The Imputer Builder)
Python
def train_and_save_imputer(df, target_cols, save_path="imputer_models.joblib"):
    """
    Trains regressors for specific columns and saves them as a dictionary.
    """
    imputers = {}
    
    # Use a clean copy for training (drop rows where predictors are missing)
    clean_df = df.dropna(subset=[c for c in df.columns if c not in target_cols])
    
    for col in target_cols:
        print(f"Training imputer for: {col}...")
        
        # Mask only the rows where this specific target is NOT missing
        mask = clean_df[col].notnull()
        X = clean_df[mask].drop(columns=target_cols)
        y = clean_df.loc[mask, col]
        
        # Initialize and fit the regressor
        regressor = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
        regressor.fit(X, y)
        
        imputers[col] = regressor
        
    joblib.dump(imputers, save_path)
    print(f"Models saved to {save_path}")
    return imputers
Cell 3: Inference Function (Real-time Prediction)
Python
def predict_missing_values(user_input_dict, model_path="imputer_models.joblib"):
    """
    Predicts missing values for a single user profile using saved models.
    """
    imputers = joblib.load(model_path)
    user_df = pd.DataFrame([user_input_dict])
    
    for col, model in imputers.items():
        # Check if the specific column is missing (None, NaN, or 0 if appropriate)
        if pd.isna(user_df[col].iloc[0]):
            print(f"Predicting missing value for {col}...")
            # Predict using other available features (drop the columns we are currently imputing)
            features = user_df.drop(columns=list(imputers.keys()))
            prediction = model.predict(features)[0]
            user_df[col] = prediction
            user_df[f"{col}_is_imputed"] = 1 # The Flag
        else:
            user_df[f"{col}_is_imputed"] = 0
            
    return user_df
Cell 4: Example Usage / Validation
Python
# 1. Train the model (Run this once)
# target_features = ['ap_hi', 'cholesterol']
# train_and_save_imputer(df, target_cols=target_features)

# 2. Simulate a new user with missing data
user_data = {'age': 50, 'weight': 80, 'ap_hi': np.nan, 'cholesterol': None}
complete_data = predict_missing_values(user_data)

print("\nFinal Input for Classifier:")
print(complete_data)
Why this architecture is best for you:
Separation of Concerns: You don't have to re-train the model every time a user makes a request. The training happens once in an offline pipeline.

Model Persistence: joblib stores the logic inside the saved file, allowing you to deploy it in a web app or API effortlessly.

Real-time Prediction: When the user leaves a field blank, the model instantly fills it using the "learned intelligence" from your original dataset, ensuring the downstream FT-Transformer receives a complete input.

By using this approach, you are effectively creating a Predictive Imputation Pipeline. It treats missing values as a downstream task where the system "infers" the missing clinical data, which is mathematically superior to simple mean-substitution.