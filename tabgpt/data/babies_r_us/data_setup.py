from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

from IPython import embed


class BabiesData(DataFrameLoader):
    def __init__(self, task_description='babies r us products'):
        super().__init__(task_description)

    def setup(self, last_nrows=50_000):

        df = pd.read_csv(os.path.join(self.current_dir, 'babies_r_us.csv'))
     

        df['weight'] = df['weight'].apply(self.convert_to_pounds)
        df['length'] = df['length'].apply(self.convert_to_inches)
        df['width'] = df['width'].apply(self.convert_to_inches)
        df['height'] = df['height'].apply(self.convert_to_inches)


        df = df.assign(is_discounted=lambda df: df.is_discounted.map({1: "yes", 0: "no"}))

        df_train, df_val = train_test_split(df, test_size=0.2, random_state=666)
        
        cat_features = ['int_id', 'ext_id', 'title', 'SKU', 'is_discounted','category', 'company_struct', 'company_free', 'brand', 'fabrics', 'colors', 'materials']
        
        num_features = ['weight', 'length', 'width', 'height']

        self.setup_scaler(num_features)
        self.scale_columns(df_train, mode='train')
        self.scale_columns(df_val)

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = num_features
        self.categorical_features = cat_features
        self.n_features = len(cat_features + num_features)
        self.set_target_column(main_target='price', additional_ones=False)

    def convert_to_pounds(self, value):
        # Handle missing values (nan)
        if pd.isna(value):
            return np.nan
        
        # Remove commas and periods from the string (except for decimal points)
        value = str(value)
        value = value.replace(',', '').replace('.', '').replace('oz', ' oz').replace('lbs', ' lbs').replace('lb', ' lb')
        
        # Initialize pounds and ounces to 0
        pounds = 0.0
        ounces = 0.0
        
        # Find pounds
        pound_match = re.search(r'(\d+(\.\d+)?)\s*lb', value)
        if pound_match:
            pounds = float(pound_match.group(1))
        
        # Find ounces
        ounce_match = re.search(r'(\d+(\.\d+)?)\s*oz', value)
        if ounce_match:
            ounces = float(ounce_match.group(1))
        
        # Convert ounces to pounds and add to pounds
        pounds += ounces / 16.0
        
        return pounds
        
    def convert_to_inches(self, value):
        # Handle missing values
        if pd.isna(value):
            return np.nan
        
        # Remove any non-numeric values (catch words, symbols)
        if not re.search(r'\d', value):
            return np.nan
        
        # Clean up the string (remove unwanted characters like ", in, etc.)
        value = re.sub(r'[^0-9.]', '', value)
        
        # If value is empty after removing non-numeric characters, set to NaN
        if value == '':
            return np.nan
        
        # Convert to float
        try:
            return float(value)
        except ValueError:
            return np.nan


