import re
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, order):
        self.column_name = column_name
        self.order = order

    def fit(self, X, y=None):
        unique_values = X[self.column_name].unique()
        self.mapping = {val: i for i, val in enumerate(self.order) if val in unique_values}
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column_name] = X[self.column_name].map(self.mapping)
        return X

    def set_output(self, transform=None):
        return self
    
class CustomCategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_convert=None):
        self.columns_to_convert = columns_to_convert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_convert is not None:
            for col in self.columns_to_convert:
                X[col] = X[col].astype("category")
        return X