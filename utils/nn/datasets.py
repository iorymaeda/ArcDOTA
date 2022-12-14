import random
from typing import Literal

import tqdm
import pandas as pd
from torch.utils.data import Dataset as D

from ..base import ConfigBase
from ..time_series import PrematchTSCollector


class PrematchDataset(ConfigBase, D):
    def __init__(self, corpus: pd.DataFrame, indexes: pd.Index, evaluate_tokenize:bool=True, y_output: Literal['binary', 'crossentropy']='binary', mask_type: str = 'bool'):
        """TODO: Fill this out with more details later

        Args:
            - corpus: pd.DataFrame - the whole (or only with two team games if it inference) 
                corpus with games in ascending chronological order
                `corpus` >= `indexes`
            - indexes: pd.Index - corpus indexes with games to output
                `indexes` <= `corpus`
        """
        self.collector = PrematchTSCollector(mask_type=mask_type, y_output=y_output)

        
        self.corpus = corpus
        self.indexes = indexes
        self.evaluate_tokenize = evaluate_tokenize


        self.cache = []

        self.config = None
        self.config = self._get_config('features')
        self.__generated = False


    @property
    def ready(self):
        return self.__generated


    def build(self) -> 'PrematchDataset':
        if not self.cache:
            for index in tqdm.tqdm(self.indexes):
                sample = self.generate_sample(index)
                if sample and (sample['r_window']['seq_len'] >= self.config['league']['window_min_size'] and 
                    sample['d_window']['seq_len'] >= self.config['league']['window_min_size']):
                    self.cache.append(sample)

        self.__generated = True
        return self


    def __len__(self):
        return len(self.cache)


    def __getitem__(self, idx) -> dict:
        assert self.__generated
        return self.cache[idx]


    def generate_sample(self, index, corpus=None) -> dict|None:
        corpus = self.corpus if corpus is None else corpus

        # Previous games
        games = corpus[:index]
        # Current game
        anchor = corpus[index:index+1]

        _anchor = self.collector.tokenizer.tokenize(anchor, teams=True, players=True)
        _anchor = self.collector.tokenizer.evaluate(_anchor)
        if self.evaluate_tokenize and len(_anchor) == 0: return None

        return self.collector.collect_windows(
            games=games, anchor=anchor, tokenize=True, 
        )


