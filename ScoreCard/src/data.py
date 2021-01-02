import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class Dataset(object):
    
    def __init__(self, X, y):
        assert len(X) == len(y), "The features' length does not align with targets' length"
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
       
        return self.X[index], self.y[index]
    
    @staticmethod
    def from_dataframe_with_feature_cols(dataframe, features_cols, target_col):
        X = dataframe[features_cols].values
        y = dataframe[target_col].values
        
        return Dataset(X, y)
    @staticmethod
    def from_dataframe_with_feature_cols(dataframe, features_cols, target_col):
        X = dataframe[features_cols].values
        y = dataframe[target_col].values
        
        return Dataset(X, y)
    @staticmethod
    def from_dataframe(dataframe, target_col):
        columns = list(dataframe.columns)
        columns.remove(target_col)
        X = dataframe[columns].values
        y = dataframe[target_col].values
        
        return Dataset(X, y)

def split_dataset(dataset, test_size = 0.2):
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, test_size = test_size, stratify = dataset.y)
    X_train, y_train = dataset[train_indices]
    
    X_test, y_test = dataset[test_indices]
    
    return Dataset(X_train, y_train), Dataset(X_test, y_test)
    
