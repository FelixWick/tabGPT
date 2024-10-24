from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import datetime as dt
import os
import holidays
import json
from pymongo import MongoClient
from sklearn.preprocessing import OrdinalEncoder
from IPython import embed


class MexicoData(DataFrameLoader):
    def __init__(self, task_description='Mexico'):
        super().__init__(task_description)

    def setup(self, testset = False):

        # Load data
        # Data source: https://www.kaggle.com/datasets/martinezjosegpe/grocery-store
        db_url = os.getenv("MONGO_LOCAL_URI")
        client = MongoClient(db_url)

        # Import df_tickets
        db = client['mexico_en']
        dbcol = db[ 'TICKETS_DETAIL']
        data = dbcol.find(
            {},
            {"IDProduct": 1, "Price": 1, "DetQuantityProds": 1, "DateOperation": 1, "_id": 0}
            ).batch_size(10000)

        df_tickets = pd.DataFrame(data)
        colname_dict = {
                    "IDProduct": "product_id",
                    "Price": "price",
                    "DetQuantityProds": "quantity",
                    "DateOperation": "datetime"
                }
        df_tickets.rename(columns=colname_dict, inplace=True)

        # Import df_products
        db = client['mexico_en']
        dbcol = db[ 'PRODUCTS']
        data = dbcol.find(
            {},
            {"IDProduct": 1, "DescProd": 1, "ShortName": 1, "Comments": 1, "_id": 0}
            ).batch_size(10000)

        df_products = pd.DataFrame(data)
        colname_dict = {
                    "IDProduct": "product_id",
                    "DescProd": "product description",
                    "ShortName": "product name"
                }
        df_products.rename(columns=colname_dict, inplace=True)
        df_products = df_products.drop(columns='Comments')

        # Add the aggregation, and then merge the 2 tables
        # Delete "money" entry
        df_tickets = df_tickets.query('product_id != "2470"')

        # Convert quantity to float for arithmetic operations
        df_tickets['quantity'] = df_tickets['quantity'].astype(str).astype(float)
        df_tickets['price'] = df_tickets['price'].astype(str).astype(float)

        # Specify train/ test period
        date_start_EWMA = pd.to_datetime('2020-10-01')
        date_start = pd.to_datetime('2021-10-01')
        date_split = pd.to_datetime('2022-07-01')
        date_end = pd.to_datetime('2022-09-30')

        # Aggregrate to daily
        df_final = self.aggregate_fillna(df_tickets, df_products)

        # Debug
        # df_final[['date', 'product_id']].duplicated().any()

        # Drop due to date (to 1 year earlier to allow for correct EWMA value from begining of train)
        df_final = df_final[df_final["date"] >= date_start_EWMA.date()].reset_index(drop=True)  # ~95000000 to ~20000000

        # Remove those with zero sales throughout the date period between date_start and date_split
        df_final = self.remove_zero_sales(df_final, date_start, date_split)   # to ~5000000

        # Obtain date features
        df_final = self.get_date_feature(df_final, country="MX")

        # Obtain EWMA features
        ewma_groups = ["product name", "day_of_week"]
        df_final = self.ewma_calculation(df_final, ewma_groups, "quantity", 0.15, 1)

        # Drop due to date: to the actual time range
        df_final = df_final[df_final["date"] >= date_start].reset_index(drop=True)   # to ~2600000

        # Split data
        df_train = df_final.query('@date_start <= date < @date_split').reset_index(drop=True)
        df_val = df_final.query('@date_split <= date <= @date_end').reset_index(drop=True)
        
        # Setting of categorical and numerical features
        categorical_features = [
            "product description",
            "product name",
            "holiday_name",
            "month",
            "year",
            "day_of_week"
        ]
        numerical_features = [
            "average daily price",
            "quantity",
            "past quantity"
        ]

        # Scaling
        self.setup_scaler(numerical_features)
        self.scale_columns(df_train, mode='train')
        self.scale_columns(df_val)

        # Drop columns
        df_train = df_train.drop(columns=['product_id'])
        df_val = df_val.drop(columns=['product_id'])

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.set_target_column(main_target='quantity', additional_ones=False)


    # Function that aggregrates to daily, fill in those with 0 sales by 0
    # When creating the matrix, use only those products that have >0 quantity throughout the period
    def aggregate_fillna(self, df_transaction, df_product):
        # Ensure datetime column is in correct datetime format
        df_transaction['datetime'] = pd.to_datetime(df_transaction['datetime'])

        # Extract date from datetime
        df_transaction['date'] = df_transaction['datetime'].dt.date

        # Generate all unique dates from the transaction data
        all_dates = pd.date_range(start=df_transaction['date'].min(), end=df_transaction['date'].max()).date

        # Get all unique products
        all_products = df_product['product_id'].unique()

        # Generate production of quantity and price
        df_transaction['quantity_price_mul'] = df_transaction['quantity'] * df_transaction['price']

        # Create a DataFrame with all date and product combinations
        date_product_combinations = pd.MultiIndex.from_product([all_dates, all_products], names=['date', 'product_id']).to_frame(index=False)

        # Aggregate the transaction data
        daily_summary_quantity = df_transaction.groupby(['date', 'product_id'])['quantity'].sum().reset_index()
        daily_summary_quantity_price_mul = df_transaction.groupby(['date', 'product_id'])['quantity_price_mul'].sum().reset_index()

        # Merge aggregated data with comprehensive date-product combinations
        df_merged = pd.merge(date_product_combinations, daily_summary_quantity, how='left', on=['date', 'product_id'])
        df_merged = pd.merge(df_merged, daily_summary_quantity_price_mul, how='left', on=['date', 'product_id'])

        # Obtain daily average price
        df_merged['average daily price'] = df_merged['quantity_price_mul'] / df_merged['quantity']
        df_merged = df_merged.drop(columns='quantity_price_mul')

        # Fill NaN quantity values with 0
        df_merged['quantity'] = df_merged['quantity'].fillna(0)

        # Fill NaN average daily price values with that of the first available previous or following value for the same product
        # THIS MAY LEAD TO LEAKAGE ISSUE (WE USE FUTURE PRICE INFORMATION THAT IS INFERRED FROM FUTURE SALES)
        # IMPLEMENTED NEVERTHELESS AS WE ASSUME WE KNOW THE PRICE AT ACTUAL IMPLEMENTATION
        df_merged['average daily price'] = df_merged.groupby('product_id')['average daily price'].ffill().bfill()

        # Merge with product names if required
        df_final = pd.merge(df_merged, df_product, left_on='product_id', right_on='product_id', how='left')

        return df_final
    

    # Remove data with zero sales throughout specified period
    def remove_zero_sales(self, df, date_start, date_end):
        # Filter the data between date_start and date_end
        df_sales_period = df[(df['date'] >= date_start.date()) & (df['date'] < date_end.date())].reset_index(drop=True)
        # df_sales_period = df.query('@date_start <= date < @date_end').reset_index(drop=True)

        # Obtain total sales per product
        total_sales_per_product = df_sales_period.groupby('product_id')['quantity'].sum()

        # Obtain products with sales
        products_with_sales = total_sales_per_product[total_sales_per_product > 0].index

        # Filter out products with zero sales
        df_final = df[df['product_id'].isin(products_with_sales)].reset_index(drop=True)

        return df_final
    

    # Add date-relateed features
    def get_date_feature(self, df_data, country):
        start_date = df_data.date.min()- dt.timedelta(1)
        end_date = df_data.date.max()
        years = list(range(start_date.year, end_date.year + 1, 1))
        mx_holiday = holidays.country_holidays(country, years=years, language="en_US")

        # Add holiday name
        df_date = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date)})
        df_date["holiday_name"] = (
            df_date["date"]
            .apply(lambda x: mx_holiday[x] if (x in mx_holiday) else "not holiday")
        )

        # # Add holiday feature
        # df_data["holiday"] = (
        #     df_data["date"]
        #     .apply(lambda x: "yes" if (x in mx_holiday) else "no")
        # )

        df_date["month"] = df_date["date"].dt.month_name(locale="en_US.UTF-8")
        df_date["year"] = df_date["date"].dt.year
        df_date["day_of_week"] = df_date["date"].dt.day_name()

        df_data['date'] = pd.to_datetime(df_data['date'])
        df_date['date'] = pd.to_datetime(df_date['date'])
        df_output = df_data.merge(df_date, how='left', on='date')

        return df_output
    

    # Extract EWMA features
    def ewma_calculation(self, df, group_cols, col, alpha, horizon):
        df.sort_values(["date"], inplace=True)
        df_grouped = df.groupby(group_cols, group_keys=False)
        df["past quantity"] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        return df


if __name__ == '__main__':
    mexico = MexicoData()
    mexico.setup()

    print(mexico.n_features)
    print(mexico.categorical_features)
    print(mexico.numerical_features)
    embed()