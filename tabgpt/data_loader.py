from dataclasses import dataclass, field
import pandas as pd
import inspect
from typing import List, Optional
import os
from sklearn.preprocessing import QuantileTransformer
import logging
import numpy as np


@dataclass
class DataFrameLoader:
    task_description: str
    numerical_features: List[str] = field(init=False, default_factory=list) 
    categorical_features: List[str] = field(init=False, default_factory=list)
    target_column: List[str] = field(init=False, default=None)
    main_target: str = field(init=False, default=None)
    n_features: int = field(init=False, default=None)
    mode: List[str] = field(init=False, default='train')
    df_train: pd.DataFrame = field(init=False, default=None) 
    df_val: Optional[pd.DataFrame] = field(init=False, default=None) 
    df_test: Optional[pd.DataFrame] = field(init=False, default=None)
    

    def __post_init__(self):
        current_class = self.__class__
        class_file = inspect.getfile(current_class)
        self.current_dir = os.path.dirname(os.path.abspath(class_file))
        self.name = os.path.basename(self.current_dir)
    

    def setup(self, *args, **kwargs) -> None:
        pass

    def df(self):
        if self.mode == 'train':
            return self.df_train
        if self.mode == 'val':
            if self.df_val is None:
                raise ValueError('No validation set available')
            return self.df_val
        if self.mode == 'test':
            if self.df_test is None:
                raise ValueError('No test set available')
            return self.df_test
        
    def setup_scaler(self, numerical_features: List[str]):
        num_info = {}
        for num_feature in numerical_features:
                qt = QuantileTransformer()
                num_info[num_feature] = qt

        self.scaler = num_info
    
    def scale_columns(self, df, mode = None):
        for num_feature, scaler in self.scaler.items():
            df[num_feature] = df[num_feature].astype(float)
            if mode == 'train':
                df[num_feature] = scaler.fit_transform(df[[num_feature]])
            else:
                df[num_feature] = scaler.transform(df[[num_feature]])
        return df
    
    def set_target_column(self, main_target,additional_ones=True):
        self.main_target = main_target
        self.target_column = [main_target]
        if additional_ones:
            additional_ones = [t for t in self.numerical_features if t!=main_target]
            for additional_target in additional_ones:
                self.target_column.append(additional_target)
        logging.info(f'{self.name} - Main target: {main_target}; additional targets: {additional_ones}')

    def append_empty_target(self, df):
        df[self.main_target] = np.ones(len(df))
        return df
    
    def remove_feature(self, f, type='numerical'):
        if type == 'numerical':
            del self.scaler[f]
            self.numerical_features.remove(f)
        else:
            self.categorical_features.remove(f)
        self.n_features -= 1
       
