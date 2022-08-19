from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


from .base import ConfigBase, SaveLoadBase


METHODS = Literal['minmax', 'minmax2', 'standart', 'box-cox', 'yeo-johnson']
MODES = Literal['teams', 'players', 'both']


class DotaScaler(ConfigBase, SaveLoadBase):
    """This is scaler is just for DOTA, with different modes: 
    `players`/`teams` for scaling players or teams features
    
    WARNING:
    Works only for one mode luague/public - so, now we need 
    storage 2 scalers for this modes
    
    //TODO: fix it or not
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
            self.__fitted_p = False
            self.__fitted_t = False
            

    def save(self, path: str):
        self._save(path)


    def __convert(self, dist: np.ndarray, scale=False):
        dist = dist.copy()
        if scale:
            dist = dist - dist.min() + 1e-5

        dist = dist.reshape(-1, 1)
        return dist


    def fit(self, X: pd.DataFrame, mode: MODES='teams', **kwargs) -> 'DotaScaler':
        """//TODO: describe this method"""
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
                'q 0.999': np.quantile(x, 0.999),
            }

            x[x > self.data[mode][f]['q 0.999']] = self.data[mode][f]['q 0.999']
            x[x < self.data[mode][f]['q 0.001']] = self.data[mode][f]['q 0.001']
            
            bc_pt = PowerTransformer(method='box-cox')
            bc_pt.fit(self.__convert(x, scale=True))

            yj_pt = PowerTransformer(method='yeo-johnson')
            yj_pt.fit(self.__convert(x, scale=False))

            self.data[mode][f]['box-cox'] = bc_pt
            self.data[mode][f]['yeo-johnson'] = yj_pt

        if mode == 'teams':
            self.__fitted_t = True
        elif mode == 'players':
            self.__fitted_p = True
        return self 
            

    def transform(self, X: pd.DataFrame, method: METHODS, mode: MODES='teams', **kwargs):
        """Apply a transform to data: power-transform, standartization, scaling

        ----------------------------------------------------------------------
        Power transforms are a family of parametric, monotonic transformations
        that are applied to make data  more Gaussian-like. This is  useful for
        modeling issues related to heteroscedasticity (non-constant variance),
        or other situations where normality is desired.
        
        Avaibale power transforms: `box-cox`, `yeo-johnson`
        Box-Cox requires input data to be strictly positive, while Yeo-Johnson
        supports both positive or negative data.

        ----------------------------------------------------------------------
        Min-max transform  features by  scaling each feature to a given  range

        This estimator  scales  and  translates each feature individually such 
        that it is in the given range 
        If `minmax` features will be scaled in range [0, 1]
        If `minmax2` features will be scaled in range [-1, 1]
        If `minmax2` and provided arguments `a` & `b` features will be scaled in range [`a`, `b`]

        ----------------------------------------------------------------------
        //TODO: write about standart transform
        """
        # ------------------------------ #
        assert method in ['minmax', 'minmax2', 'standart', 'box-cox', 'yeo-johnson'], 'Normalization type is not correct'
        assert mode in ['players', 'teams', 'both']
        assert type(X) is pd.DataFrame

        # ------------------------------ #
        X = X.copy()
        if mode == 'players':
            assert self.__fitted_p
            sides = self.RADIANT_SIDE + self.DIRE_SIDE
            
        elif mode == 'teams':
            assert self.__fitted_t
            sides = ['r', 'd']

        elif mode == 'both':
            assert self.__fitted_p
            assert self.__fitted_t
            X = self.transform(X=X, method=method, mode='teams', **kwargs)
            X = self.transform(X=X, method=method, mode='players', **kwargs)
            return X

        for f in self.FEATURES:
            saved_feature = self.data[mode][f]
            for side in sides:
                # ------------------------------ #
                x = X[f"{side}_{f}"].values.astype('float32')

                # ------------------------------ #
                # convert outliers to normal values
                x[x > saved_feature['q 0.999']] = saved_feature['q 0.999']
                x[x < saved_feature['q 0.001']] = saved_feature['q 0.001']
                
                # ------------------------------ #
                # normilize
                if method == 'minmax':
                    x = (x - saved_feature['q 0.001'])/(saved_feature['q 0.999'] - saved_feature['q 0.001'])
                        
                elif method == 'minmax2':
                    # min max in [a, b]
                    if 'a' in kwargs and 'b' in kwargs:
                        a, b = kwargs['a'], kwargs['b']
                    else:
                        a, b = -1, 1

                    x = (b-a)*(x - saved_feature['q 0.001'])/(saved_feature['q 0.999'] - saved_feature['q 0.001']) + a
                    
                elif method == 'standart':
                    x = (x - saved_feature['mean'])/saved_feature['std']
                
                elif method == 'box-cox':
                    pt: PowerTransformer = saved_feature['box-cox']
                    x = pt.transform(self.__convert(x, scale=True))
                    x = x[:, 0]

                elif method == 'yeo-johnson':
                    pt: PowerTransformer = saved_feature['yeo-johnson']
                    x = pt.transform(self.__convert(x, scale=False))
                    x = x[:, 0]

                # ------------------------------ #
                X[f"{side}_{f}"] = x

        return X


