from dataclasses import dataclass, field
import pandas as pd
import inspect
from typing import List, Optional
import os

@dataclass
class DataFrameLoader:
    task_description: str
    numerical_features: List[str] = field(init=False, default_factory=list) 
    categorical_features: List[str] = field(init=False, default_factory=list)
    target_column: str = field(init=False, default=None)
    n_features: int = field(init=False, default=None)
    mode: List[str] = field(init=False, default='train')
    df_train: pd.DataFrame = field(init=False, default=None) 
    df_val: Optional[pd.DataFrame] = field(init=False, default=None) 
    df_test: Optional[pd.DataFrame] = field(init=False, default=None)
    test_target_column: str = field(init=False,default=None)

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