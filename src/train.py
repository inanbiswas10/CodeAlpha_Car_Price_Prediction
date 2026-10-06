import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from data_loader import load_and_clean_car_data
from features import engineer_car_features


def build_and_evaluate_models (data_path,model_save_path):

    # Loads car pricing dataset, applies feature engineering, constructs 
    # preprocessing & regression pipelines, evaluates models and saves the best model.

    # 1) Load and Engineer Features

    df_raw = load_and_clean_car_data (data_path)
    df = engineer_car_features (df_raw)

    # 2) Separate Target and Feature Columns
    # Note: 'selling_price' is our target variable (in Lakhs)
    # We drop 'price_difference' and 'value_retained_pct' to avoid target leakage

    X = df.drop (columns = ['selling_price','car_name','price_difference','value_retained_pct','brand'],errors = 'ignore')
    y = df ['selling_price']

    # 3) Identify Categorical and Numerical Columns

    categorical_cols = ['fuel_type','seller_type','transmission']
    numerical_cols = ['year','present_price','kms_driven','owner','vehicle_age','is_luxury_brand','kms_per_year']

    # Filter columns actually present in X

    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numerical_cols = [c for c in numerical_cols if c in X.columns]

    # 4) Construct ColumnTransformer for Preprocessing

    preprocessor = ColumnTransformer (
        transformers = [
            ('num',StandardScaler (),numerical_cols),
            ('cat',OneHotEncoder (drop = 'first',handle_unknown = 'ignore'),categorical_cols)])

    # 5) Train / Test Split

    X_train,X_test,y_train,y_test = train_test_split (X,y,test_size = 0.2,random_state = 42)

    # 6) Candidate Models Evaluation Matrix

    candidate_models = {
        'Linear Regression': LinearRegression (),
        'Ridge Regression': Ridge (alpha = 1.0),
        'Random Forest': RandomForestRegressor (n_estimators = 100,random_state = 42),
        'Gradient Boosting': GradientBoostingRegressor (n_estimators = 100,random_state = 42),
        'XGBoost': XGBRegressor (n_estimators = 100,random_state = 42,learning_rate = 0.1)}

    results = {}
    best_r2 = -float ('inf')
    best_pipeline = None
    best_model_name = ""

    print ("="*60)
    print ("MODEL PERFORMANCE EVALUATION BENCHMARK")
    print ("="*60)

    for name,model in candidate_models.items ():
        pipeline = Pipeline (steps = [
            ('preprocessor',preprocessor),
            ('regressor',model)])

        # Fit model pipeline

        pipeline.fit (X_train,y_train)

        # Predictions

        y_pred = pipeline.predict (X_test)

        # Performance Metrics

        mae = mean_absolute_error (y_test,y_pred)
        rmse = np.sqrt (mean_squared_error (y_test,y_pred))
        r2 = r2_score (y_test,y_pred)

        results [name] = {'MAE': mae,'RMSE': rmse,'R2': r2}

        print (f"[{name}]")
        print (f"  R2 Score: {r2:.4f}")
        print (f"  MAE     : {mae:.4f} Lakhs")
        print (f"  RMSE    : {rmse:.4f} Lakhs\n")

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline
            best_model_name = name

    print ("-"*60)
    print (f"TOP PERFORMING MODEL: {best_model_name} (R2 = {best_r2:.4f})")
    print ("-"*60)

    # 7) Save Best Model Pipeline

    os.makedirs (os.path.dirname (model_save_path),exist_ok = True)
    joblib.dump (best_pipeline,model_save_path)
    print (f"Saved optimized model pipeline to: {model_save_path} !!")

    return best_model_name,best_r2

if __name__ == "__main__":
    
    data_file = os.path.join ("data","car_data.csv")
    model_file = os.path.join ("models","car_price_model.pkl")
    build_and_evaluate_models (data_file, model_file)