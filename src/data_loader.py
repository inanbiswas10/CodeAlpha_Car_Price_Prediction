import os
import pandas as pd
import numpy as np

def load_and_clean_car_data (file_path):

    # Loads raw car pricing data, cleans column headers, handles missing values,
    # and standardizes column names and data types.

    if not os.path.exists (file_path):
        raise FileNotFoundError (
            f"Dataset file not found at: {file_path} !! Ensure 'car_data.csv' is placed inside 'data/'.")

    # Read CSV file

    df = pd.read_csv (file_path)

    # Standardize column headers

    df.columns = df.columns.str.strip ().str.lower ().str.replace (' ','_')

    # Explicit column rename mapping

    rename_dict = {
        'car_name': 'car_name',
        'year': 'year',
        'selling_price': 'selling_price',
        'present_price': 'present_price',
        'driven_kms': 'kms_driven',
        'fuel_type': 'fuel_type',
        'selling_type': 'seller_type',
        'transmission': 'transmission',
        'owner': 'owner'
    }
    df.rename (columns = rename_dict,inplace = True)

    # Drop missing target rows if any exist

    if 'selling_price' in df.columns:
        df.dropna (subset = ['selling_price'],inplace = True)

    # Strip whitespaces from string/object columns (compatible with Pandas 2 & 3)

    string_cols = df.select_dtypes (include = ['object','string','category']).columns
    for col in string_cols:
        df [col] = df [col].astype (str).str.strip ()

    return df

if __name__ == "__main__":
    data_path = os.path.join ("data","car_data.csv")
    try:
        df = load_and_clean_car_data (data_path)
        print ("Data loaded successfully without warnings !!")
        print (f"Dataset Shape: {df.shape}")
        print ("\nCleaned Column Names:")
        print (list(df.columns))
    except Exception as e:
        print (f"Error loading dataset: {e} !!")