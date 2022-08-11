import numpy as np
import pandas as pd

from .base import ConfigBase


class BaseScaler(ConfigBase):
    def __init__(self, features):
        self.FEATURES = features
        # Stores statistics per feature
        self.data = {}
            

    def fit(self, X: pd.DataFrame):
        assert type(X) is pd.DataFrame

        X = X.copy()
        for f in self.FEATURES:
            x = pd.concat([X[f'r_{f}'], X[f'd_{f}']])
            x = x.values.astype('float32')
            self.data[f] = {
                'min': x.min(),
                'max': x.max(),
                'mean': x.mean(),
                'std': x.std(),
                'median': np.median(x),
                'q 0.001': np.quantile(x, 0.001),
                'q 0.999': np.quantile(x, 0.999)} 
            

    def transform(self, X: pd.DataFrame, t: str, **kwargs):
        assert t in ['minmax', 'minmax2', 'standart'], 'Normalization type is not correct'
        assert type(X) is pd.DataFrame

        X = X.copy()
        for f in self.FEATURES:
            for side in ['r', 'd']:
                # ------------------------------ #
                x = X[f"{side}_{f}"].astype('float32')

                # ------------------------------ #
                # convert outliers to normal values
                x[x > self.data[f]['q 0.999']] = self.data[f]['q 0.999']
                x[x < self.data[f]['q 0.001']] = self.data[f]['q 0.001']
                
                # ------------------------------ #
                # normilize
                if t == 'minmax':
                    x = (x - self.data[f]['min'])/(self.data[f]['max'] - self.data[f]['min'])
                        
                elif t == 'minmax2':
                    # min max in [a, b]
                    if 'a' in kwargs and 'b' in kwargs:
                        a, b = kwargs['a'], kwargs['b']
                    else:
                        a, b = -1, 1

                    x = (b-a)*(x - self.data[f]['min'])/(self.data[f]['max'] - self.data[f]['min']) + a
                    
                elif t == 'standart':
                    x = (x - self.data[f]['mean'])/self.data[f]['std']
                    
                # ------------------------------ #
                X[f"{side}_{f}"] = x

        return X