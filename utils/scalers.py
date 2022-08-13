from typing import Literal

import numpy as np
import pandas as pd

from .base import ConfigBase, SaveLoadBase


T = Literal['minmax', 'minmax2', 'standart']
MODE = Literal['teams', 'players']


class BaseScaler(ConfigBase, SaveLoadBase):
    """This is scaler is just for DOTA, with different modes: 
    `players`/`teams` for scaling players or teams features
    
    WARNING:
    Works only for one mode luague/public //TODO: fix it or not
    So, now we need storage 2 scalers for this modes
    """

    def __init__(self, features:list=None, path:str=None):
        if path is None and features is None:
            raise Exception('Provide at least one argument')

        if path is not None:
            self._load(path)

        elif features is not None:
            self.FEATURES = features
            # Stores statistics per feature
            self.data = {
                'players': {},
                'teams': {},
            }
            

    def save(self, path: str):
        self._save(path)


    def fit(self, X: pd.DataFrame, mode: MODE='teams', **kwargs) -> 'BaseScaler':
        assert type(X) is pd.DataFrame

        sides = self.RADIANT_SIDE + self.DIRE_SIDE if mode == 'players' else ['r', 'd']

        X = X.copy()
        for f in self.FEATURES:
            x = pd.concat([X[f'{side}_{f}'] for side in sides], axis=0)
            x = x.values.astype('float32')
            self.data[mode][f] = {
                'min': x.min(),
                'max': x.max(),
                'mean': x.mean(),
                'std': x.std(),
                'median': np.median(x),
                'q 0.001': np.quantile(x, 0.001),
                'q 0.999': np.quantile(x, 0.999)}

        return self 
            

    def transform(self, X: pd.DataFrame, t: T, mode: MODE='teams', **kwargs):
        assert t in ['minmax', 'minmax2', 'standart'], 'Normalization type is not correct'
        assert type(X) is pd.DataFrame

        sides = self.RADIANT_SIDE + self.DIRE_SIDE if mode == 'players' else ['r', 'd']

        X = X.copy()
        for f in self.FEATURES:
            saved_feature = self.data[mode][f]
            for side in sides:
                # ------------------------------ #
                x = X[f"{side}_{f}"].astype('float32')

                # ------------------------------ #
                # convert outliers to normal values
                x[x > saved_feature['q 0.999']] = saved_feature['q 0.999']
                x[x < saved_feature['q 0.001']] = saved_feature['q 0.001']
                
                # ------------------------------ #
                # normilize
                if t == 'minmax':
                    x = (x - saved_feature['q 0.001'])/(saved_feature['q 0.999'] - saved_feature['q 0.001'])
                        
                elif t == 'minmax2':
                    # min max in [a, b]
                    if 'a' in kwargs and 'b' in kwargs:
                        a, b = kwargs['a'], kwargs['b']
                    else:
                        a, b = -1, 1

                    x = (b-a)*(x - saved_feature['q 0.001'])/(saved_feature['q 0.999'] - saved_feature['q 0.001']) + a
                    
                elif t == 'standart':
                    x = (x - saved_feature['mean'])/saved_feature['std']
                    
                # ------------------------------ #
                X[f"{side}_{f}"] = x

        return X