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


class MexicoDataWeekly(DataFrameLoader):
    def __init__(self, task_description='Mexico'):
        super().__init__(task_description)

    def setup(self, testset = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))        

        # Take data directly from mongodb
        # To be fixed to include more pre-processing in the future
        # Data originally from kaggle: https://www.kaggle.com/datasets/martinezjosegpe/grocery-store
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
        df_tickets = df_tickets.rename(columns={'IDProduct': 'product_id'})

        # Import df_products
        db = client['mexico_en']
        dbcol = db[ 'PRODUCTS']
        data = dbcol.find(
            {},
            {"IDProduct": 1, "DescProd": 1, "ShortName": 1, "Comments": 1, "_id": 0}
            ).batch_size(10000)

        df_products = pd.DataFrame(data)
        df_products = df_products.rename(columns={'IDProduct': 'product_id'})

        # Type change
        df_tickets['date'] = pd.to_datetime(pd.to_datetime(df_tickets['DateOperation']).dt.date)
        df_tickets = df_tickets.copy()
        df_tickets['DetQuantityProds'] = df_tickets['DetQuantityProds'].astype(str).astype(float)
        df_tickets['Price'] = df_tickets['Price'].astype(str).astype(float)

        # Delete specific data
        # To be reviewed
        df_tickets = df_tickets.query('product_id != "2470"')

        # Screen time period
        DATA_START_DATE = pd.to_datetime('2014-02-01') # data start date
        BASE_DATE = pd.to_datetime('2022-04-01')
        DATA_LAST_DATE = pd.to_datetime('2022-09-30')
        max_train_data_period = 3103

        save_key = 'None' # mongodb, parquet, None
        seed = 42
        np.random.seed(seed)

        # Product numnber
        NUM_PROD = 1000

        # Product screen settings
        product_key = 'sales_period_top_half_random' # 'top_frequency', 'top_frequency_half', 'top_quantity_sum', 'top_quantity_mean', 'random'
        '''
        'top_frequency_half' : 'Randomly_sampling_from_top_half_of_frequency'  
        'top_frequency' : 'Sampling_the_top_half_of_frequency'
        'top_quantity_sum' : 'Sampling_of_top_total_quantity' 
        'top_quantity_mean' : 'Sampling_of_top_mean_quantity_per_day'
        'random' : 'random'
        sales_period_top5
        sales_period_top_half_random
        # 'Randomly_sampling_from_top_half_of_frequency', 'Sampling_the_top_half_of_frequency', 'Sampling_of_top_total_quantity', 'Sampling_of_top_mean_quantity_per_day'
        '''

        # Time period
        Product_sampled_period = 365
        TRAIN_DATA_PERIOD = 365 * 9  # max_train_data_period = 3103 < 365 * 9
        TEST_DATA_PERIOD = 60  #30

        # define train, test period
        if 0 < TRAIN_DATA_PERIOD < max_train_data_period:
            TRAIN_DATA_START_DATE = BASE_DATE - dt.timedelta(days=TRAIN_DATA_PERIOD)
        elif TRAIN_DATA_PERIOD >= max_train_data_period:
            # use all data
            print('use all data')
            TRAIN_DATA_START_DATE = DATA_START_DATE
        else:
            assert False, 'miss train data period'
        TEST_DATA_LAST_DATE = DATA_LAST_DATE

        # define Product_sampled_period
        Product_sampled_period_start_date = BASE_DATE - dt.timedelta(days=Product_sampled_period)
        print(Product_sampled_period_start_date, TRAIN_DATA_START_DATE, TEST_DATA_LAST_DATE)

        data_config = {
            'NUM_PROD' : NUM_PROD,
            'product_key' : product_key,
            'BASE_DATE' : BASE_DATE,
            'DATA_START_DATE' : DATA_START_DATE,
            'TEST_DATA_LAST_DATE' : TEST_DATA_LAST_DATE,
            'Product_sampled_period_start_date' : Product_sampled_period_start_date,
            'seed' : 42,
        }

        # Product sampling
        def product_sampling(df_data, data_config, df_tickets, BASE_DATE):
            
            DATA_START_DATE = data_config['DATA_START_DATE']
            TEST_DATA_LAST_DATE = data_config['TEST_DATA_LAST_DATE']
            Product_sampled_period_start_date = data_config['Product_sampled_period_start_date']
            product_key = data_config['product_key']
            NUM_PROD = data_config['NUM_PROD']
            seed = data_config['seed']
            
            # Define df_product_sampled_data to narrow down on the products
            # Time span for product narrowing down could be shorter
            df_tickets_data = df_data.query('@DATA_START_DATE <= date <= @TEST_DATA_LAST_DATE')
            df_product_sampled_data = df_tickets_data.query('@Product_sampled_period_start_date <= date < @BASE_DATE')
            
            # Screen products
            print(product_key)
            np.random.seed(seed)
            if product_key == 'top_frequency':
                # Sampling_the_top_half_of_frequency
                item_all = df_product_sampled_data['product_id'].value_counts().sort_values(ascending=False).index
                item_sample = item_all[:NUM_PROD]
            elif product_key == 'top_frequency_half':
                # Randomly_sampling_from_top_half_of_frequency
                item_all = df_product_sampled_data['product_id'].value_counts().index #.sort_values(ascending=False).index
                item_all = item_all[:int(len(item_all)/2)]
                item_sample = np.random.permutation(item_all)[:NUM_PROD]
            elif product_key == 'top_quantity_sum':
                # Sampling_of_top_total_quantity
                item_all = df_product_sampled_data.groupby('product_id')['DetQuantityProds'].sum().sort_values(ascending=False).index
                item_sample = item_all[:NUM_PROD]
            elif product_key == 'top_quantity_mean':
                # Sampling_of_top_mean_quantity_per_day
                item_all = df_product_sampled_data.groupby(['date', 'product_id'])['DetQuantityProds'].sum()\
                    .reset_index().groupby(['product_id'])['DetQuantityProds'].mean().sort_values(ascending=False).index
                item_sample = item_all[:NUM_PROD]
            elif product_key == 'random':
                # random
                item_all = df_product_sampled_data['product_id'].unique()
                item_sample = np.random.permutation(item_all)[:NUM_PROD]
            elif product_key == "sales_period_top5":
                product_info = df_tickets.groupby("product_id")["date"].agg(["min", "max"])
                product_info = product_info.query(f"min < '{BASE_DATE}' & max >= '{BASE_DATE}'")
                product_info["sales_period"] = product_info["max"] - product_info["min"]
                product_info["sales_period"] = product_info["sales_period"].dt.days
                product_info = product_info.sort_values("sales_period", ascending=False)
                
                train_30 = BASE_DATE - dt.timedelta(days=31)
                train_product = df_tickets.query("@train_30 <= date <= @BASE_DATE")["product_id"].unique()
                
                product_target = set(product_info.index[:int(len(product_info) * 0.05)])
                
                product_target = list(set(train_product) & set(product_target))
                
                item_sample = list(product_target)

            elif product_key == "sales_period_top_half_random":
                product_info = df_tickets.groupby("product_id")["date"].agg(["min", "max"])
                product_info = product_info.query(f"min < '{BASE_DATE}' & max >= '{BASE_DATE}'")
                product_info["sales_period"] = product_info["max"] - product_info["min"]
                product_info["sales_period"] = product_info["sales_period"].dt.days
                product_info = product_info.sort_values("sales_period", ascending=False)
                
                train_30 = BASE_DATE - dt.timedelta(days=31)
                train_product = df_tickets.query("@train_30 <= date <= @BASE_DATE")["product_id"].unique()
                
                product_target = set(product_info.index[:int(len(product_info) * 0.50)])
                product_target = list(product_target)
                
                product_target = list(set(train_product) & set(product_target))
                
                item_sample = np.random.permutation(product_target)[:NUM_PROD]

            df_tickets_data_top_product = df_tickets_data.query('product_id in @item_sample')
            
            return df_tickets_data_top_product


        # Screen products
        df_tickets_data_top_product = product_sampling(df_tickets, data_config, df_tickets, BASE_DATE)

        # Pre-process
        df_data = df_tickets_data_top_product[['product_id', 'Price', 'DetQuantityProds', 'date']]

        # encode product
        oe_product = OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1)
        df_data['product_id_oe'] = oe_product.fit_transform(df_data['product_id'].values.reshape(-1, 1))
    

        def get_qty_price_per_week(df_data):
            """get qty and price per week and product
            """
            
            # Calculate weighted average prices per date and product
            df_data['qty_pri'] = df_data['DetQuantityProds'] * df_data['Price']               
            df_per_day = df_data.groupby(
                        ['date', 'product_id_oe'], 
                        as_index=False
                    )[['qty_pri', 'DetQuantityProds']].sum()
            
            df_per_day = df_per_day.rename(
                            columns={
                                'qty_pri' : 'sum_qty_pri', 
                                'DetQuantityProds' : 'quantity'
                                }
                        )
            
            # sum qty per week and product
            df_per_day = df_per_day.set_index('date')
            df_unique_day_week_qty = df_per_day.groupby('product_id_oe').resample('W')['quantity', 'sum_qty_pri'].sum().reset_index()
            
            pivot_df_qty_week = df_unique_day_week_qty.pivot_table(
                index='date', 
                columns='product_id_oe', 
                values='quantity', 
                aggfunc='sum', 
                fill_value=0
                )
            
            df_unique_day_week_qty['weighted_ave_price'] = (
                df_unique_day_week_qty['sum_qty_pri'] / df_unique_day_week_qty['quantity']
                ).round(3)
            
            pivot_df_price_week = df_unique_day_week_qty.pivot_table(
                index='date', 
                columns='product_id_oe', 
                values='weighted_ave_price', 
                fill_value=pd.NA
                )
            pivot_df_price_week = pivot_df_price_week.ffill().bfill()
            
            return df_unique_day_week_qty, pivot_df_qty_week, pivot_df_price_week


        # To weekly
        df_unique_day_week_qty, pivot_df_qty_week, pivot_df_price_week = get_qty_price_per_week(df_data)

        # Obtain the average quantity in the past 4 weeks
        df_month_avg_quantity = pivot_df_qty_week.rolling(4).mean().shift().stack().rename("last_month_average_quantity").reset_index()

        # Merge product ID & description
        df_X = pivot_df_price_week.stack().rename("price").reset_index()
        df_X = df_X.merge(df_month_avg_quantity, how='left', on=['date', 'product_id_oe'])
        df_X = df_X[df_X['last_month_average_quantity'].notna()]
        df_X['product_id_oe'] = df_X['product_id_oe'].astype(int)
        df_X['product_id'] = oe_product.categories_[0][df_X['product_id_oe']]
        df_X = df_X.merge(df_products, how='left', on='product_id')
        df_X['start_date'] = df_X['date'] - dt.timedelta(days=6)
        
        # Merge with actual quantity
        df_y = pivot_df_qty_week.stack().rename('quantity').reset_index()
        df_y['product_id_oe'] = df_y['product_id_oe'].astype(int)
        df_X = df_X.merge(df_y, how='left', on=['date', 'product_id_oe'])

        # Remove those with 0 data in a week
        df_X = df_X.query("date <= @TEST_DATA_LAST_DATE")

        # Timing setting
        TRAIN_DATA_PERIOD, BASE_DATE, TRAIN_DATA_START_DATE, TEST_DATA_LAST_DATE
        df_X_data = df_X.query('@TRAIN_DATA_START_DATE <= date <= @TEST_DATA_LAST_DATE').reset_index(drop=True)


        def get_date_feature(df_data, country="MX"):
            """Get date feature.

            Get date feature.

            Parameters
            ----------
            df_data : pd.DataFrame
                base dataframe.

            Returns
            -------
            df_date : pd.DtaFrame
                date feature.
            """
            start_date = df_data.date.min()- dt.timedelta(7)
            print(start_date)
            end_date = df_data.date.max()
            years = list(range(start_date.year, end_date.year + 1, 1))

            mx_holiday = holidays.country_holidays(country, years=years, language="en_US")

            df_date = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date)})
            df_date["holiday_name"] = (
                df_date["date"]
                .apply(lambda x: dt.datetime.date(x))
                .apply(lambda x: mx_holiday[x] if (x in mx_holiday) else None)
            )
            df_date["holiday"] = df_date["date"].apply(lambda x: dt.datetime.date(x)).apply(lambda x: x in mx_holiday)

            oe_holiday = OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1)
            df_date["holiday_oe"] = oe_holiday.fit_transform(df_date["holiday_name"].values.reshape(-1, 1))

            df_date["dayofweak"] = df_date["date"].dt.dayofweek.values
            df_date["dayofyear"] = df_date["date"].dt.dayofyear.values
            df_date['weekday_name'] = df_date['date'].dt.day_name()

            return df_date

        # Obtain date features
        df_date = get_date_feature(df_X_data, country="MX")

        # Weekly
        df_date.loc[df_date['holiday_name'].notna(), 'holiday_name'] = df_date['holiday_name'][df_date['holiday_name'].notna()] + ','
        df_week = df_date.set_index('date').resample('W').agg(
            week_holiday_name=("holiday_name","sum"), 
            week_holiday=("holiday","sum"), 
            weekofyear=("dayofyear","mean"),
            )
        df_week = df_week.reset_index()
        df_week['week_holiday_name'] = df_week['week_holiday_name'].replace({0:'None'})

        df_weekfeature = df_X_data.merge(df_week, how='left', on='date')

        # Save the data
        df = df_weekfeature.copy()
        df_train_full = df.query('@TRAIN_DATA_START_DATE <= date < @BASE_DATE').reset_index(drop=True)
        df_test = df.query('@BASE_DATE <= date <= @TEST_DATA_LAST_DATE').reset_index(drop=True)


        # Necessities from Ulf's code
        categorical_features = [
            "DescProd",
            "week_holiday_name",
            "date"
        ]
        numerical_features = [
            "price",
            "last_month_average_quantity"
        ]
        
        # Train: ~7 yrs, valid: ~1 yr, test: 6 months
        # Calculate total days between these dates
        total_days = (BASE_DATE - DATA_START_DATE).days
        
        # Calculate the validation period which is the latter 20% of this period
        valid_period_days = int(0.2 * total_days)
        # Calculate the start date of the validation period
        valid_start_date = BASE_DATE - pd.Timedelta(days=valid_period_days)

        df_train = df_train_full[df_train_full["date"] < valid_start_date].reset_index(drop=True)
        df_val = df_train_full[df_train_full["date"] >= valid_start_date].reset_index(drop=True)

        df_train["target"] = np.log(1 + df_train["quantity"])
        df_val["target"] = df_val["quantity"]

        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max

        # Test
        df_test["target"] = df_test["quantity"]
        df_test[numerical_features] = df_test[numerical_features] / num_max
        self.df_test = df_test

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"



if __name__ == '__main__':
    mexico = MexicoDataWeekly()
    mexico.setup()

    print(mexico.n_features)
    print(mexico.categorical_features)
    print(mexico.numerical_features)
    print(mexico.target_column)
    embed()