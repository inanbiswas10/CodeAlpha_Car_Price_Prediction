import pandas as pd
import numpy as np

def engineer_car_features (df,current_year = 2026):

    # Engineers new domain-specific features and returns preprocessed dataframe for model training.

    df_feat = df.copy ()

    # 1) Derive Vehicle Age

    if 'year' in df_feat.columns:
        df_feat ['vehicle_age'] = current_year - df_feat ['year']

    # 2) Price Differences & Retained Value Metrics

    if 'selling_price' in df_feat.columns and 'present_price' in df_feat.columns:
        df_feat ['price_difference'] = df_feat ['present_price']-df_feat ['selling_price']
        df_feat ['value_retained_pct'] = np.where (
            df_feat ['present_price'] > 0,
            (df_feat ['selling_price'] / df_feat ['present_price'])*100,0)

    # 3) Extract Brand / Brand Tier Category

    if 'car_name' in df_feat.columns:
        df_feat ['brand'] = df_feat ['car_name'].apply (lambda x: str (x).split ()[0].lower ())
        luxury_brands = ['mercedes','bmw','audi','land','jaguar','porsche','volvo','fortuner']
        df_feat ['is_luxury_brand'] = df_feat ['brand'].apply (lambda x: 1 if any (b in x for b in luxury_brands) else 0)

    # 4) Kilometer Usage Intensity per Year

    if 'kms_driven' in df_feat.columns and 'vehicle_age' in df_feat.columns:
        df_feat ['kms_per_year'] = np.where (
            df_feat ['vehicle_age'] > 0,
            df_feat ['kms_driven'] / df_feat ['vehicle_age'],
            df_feat ['kms_driven'])
    return df_feat

if __name__ == "__main__":
    
    from data_loader import load_and_clean_car_data
    import os

    data_path = os.path.join ("data","car_data.csv")
    df_raw = load_and_clean_car_data (data_path)
    df_engineered = engineer_car_features (df_raw)

    print ("Engineered Feature Matrix Head:")
    print (df_engineered [['car_name','vehicle_age','price_difference','is_luxury_brand','kms_per_year']].head ())