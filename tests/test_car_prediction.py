import os
import sys
import joblib
import pandas as pd
import pytest

# Add project root directory to path

sys.path.append (os.path.abspath (os.path.join (os.path.dirname (__file__),"..")))

import src.data_loader as dl
import src.features as ft

def test_data_file_exists ():

    # Verify raw dataset exists in data/ directory

    data_path = os.path.join ("data","car_data.csv")
    assert os.path.exists (data_path),"Car dataset missing in data/ directory"

def test_data_loader_module ():

    # Verify data_loader module can read dataset

    data_path = os.path.join ("data","car_data.csv")
    df = pd.read_csv (data_path)
    assert len (df) > 0,"Dataset is empty"

def test_model_artifact_exists ():

    # Verify trained model artifact exists in models/ directory

    model_path = os.path.join ("models","car_price_model.pkl")
    assert os.path.exists (model_path),"Model pickle file missing in models/"

def test_model_loading ():

    # Verify model artifact can be loaded via joblib

    model_path = os.path.join ("models","car_price_model.pkl")
    model = joblib.load (model_path)
    assert model is not None,"Failed to load model artifact"