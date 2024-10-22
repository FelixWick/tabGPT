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


class AmazonData(DataFrameLoader):
    def __init__(self, task_description='amazon'):
        super().__init__(task_description)

    def setup(self, testset = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))        

        # Load data
        # Data source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY
        DATA_PATH = '/'
        df_all = pd.read_csv(os.path.join(DATA_PATH, 'amazon-purchases.csv'))

        colname_dict = {
                    "Order Date": "datetime",
                    "Purchase Price Per Unit": "price",
                    "Quantity": "quantity",
                    "Title": "product name",
                    "ASIN/ISBN (Product Code)": "product_id",
                    "Category": "product category"
                }
        df_all.rename(columns=colname_dict, inplace=True)

        df_all = df_all.drop(columns=["Shipping Address State", "Survey ResponseID"])
        df_all["datetime"] = pd.to_datetime(df_all["datetime"])

        # List the date in ascending order
        df_all = df_all.sort_values(by='datetime', ascending=True).reset_index(drop=True)

        # Number of entries drop noticably after 2022-12-20
        # df_all['datetime'].value_counts().sort_index()

        # Specify train/ test period
        date_start_EWMA = pd.to_datetime('2018-01-01')
        date_start = pd.to_datetime('2019-06-01')
        date_split = pd.to_datetime('2022-06-01')
        date_end = pd.to_datetime('2022-12-01')

        # Cut between the date_start_EWMA and date_end
        df_all = df_all[(df_all['datetime'] >= date_start_EWMA) & (df_all['datetime'] <= date_end)].reset_index(drop=True)  # ~1780000

        # Aggregrate to daily
        df_final = self.aggregate_no_fill(df_all)  # ~1730000

        # Drop due to date (to 1 year earlier to allow for correct EWMA value from begining of train)
        df_final = df_final[df_final["date"] >= date_start_EWMA.date()].reset_index(drop=True)

        # Remove those with zero sales throughout the date period between date_start and date_split
        df_final = self.remove_zero_sales(df_final, date_start, date_split)   # ~1440000

        # Obtain date features
        df_final = self.get_date_feature(df_final, country="US")

        # Obtain EWMA features
        ewma_groups = ["product category", "product name", "day_of_week"]  # Not sure
        df_final = self.ewma_calculation(df_final, ewma_groups, "quantity", 0.15, 1)

        # Drop due to date: to the actual time range
        df_final = df_final[df_final["date"] >= date_start].reset_index(drop=True)   # ~1310000

        # Split data
        df_train = df_final.query('@date_start <= date < @date_split').reset_index(drop=True)
        df_test = df_final.query('@date_split <= date <= @date_end').reset_index(drop=True)

        # Normalize data
        df_train["target"] = np.log(1 + df_train["quantity"])
        df_test["target"] = df_test["quantity"]
        
        # Setting of categorical and numerical features
        categorical_features = [
            "product category",
            "product name",
            "holiday_name",
            "month",
            "year",
            "day_of_week"
        ]
        numerical_features = [
            "average daily price",
            "past target value"
        ]

        # More normalization
        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_test[numerical_features] = df_test[numerical_features] / num_max

        # Drop columns
        df_train = df_train.drop(columns=['product_id'])
        df_test = df_test.drop(columns=['product_id'])

        self.df_train = df_train
        self.df_test = df_test
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"


    # Function that aggregrates to daily, do not fill in those with 0 sales by 0
    def aggregate_no_fill(self, df_transaction):
        # Extract date from datetime
        df_transaction['date'] = df_transaction['datetime'].dt.date

        # Generate production of quantity and price
        df_transaction['quantity_price_mul'] = df_transaction['quantity'] * df_transaction['price']

        # Aggregate those with multiple sales in a day to daily transaction
        df_final = df_transaction.groupby(['date', 'product_id']).agg(
            {'quantity': 'sum', 'quantity_price_mul': 'sum', 'product name': 'first', 'product category': 'first'}).reset_index()
        
        # Obtain daily average price
        df_final['average daily price'] = df_final['quantity_price_mul'] / df_final['quantity']
        df_final = df_final.drop(columns='quantity_price_mul')

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
        df["past target value"] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        return df


if __name__ == '__main__':
    amazon = AmazonData()
    amazon.setup()

    print(amazon.n_features)
    print(amazon.categorical_features)
    print(amazon.numerical_features)
    embed()