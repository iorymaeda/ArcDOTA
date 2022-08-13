import random

import pandas as pd
from torch.utils.data import Dataset as D

from ..base import ConfigBase
from ..time_series import TSCollector


class LeagueDataset(ConfigBase, D):
    def __init__(self, corpus: pd.DataFrame, indexes: pd.Index, y_output:str, mask_type: str = 'bool'):
        """TODO: Fill this out with more details later

        Args:
            - corpus: pd.DataFrame - the whole (or only with two team games if it inference) 
                corpus with games in ascending chronological order
                `corpus` >= `indexes`
            - indexes: pd.Index - corpus indexes with games to output
                `indexes` <= `corpus`
        """
        assert y_output in ['binary', 'crossentropy']

        self.collector = TSCollector(mask_type=mask_type)

        self.corpus = corpus
        self.indexes = indexes
        self.y_output = y_output
        
        self.cache = {}

        self.config = None
        self.config = self._get_config('features')
        self.__generated = False


    @property
    def ready(self):
        return self.__generated


    def build(self) -> 'LeagueDataset':
        if not self.cache:
            cached_num = 0
            for index in self.indexes:
                sample = self.generate_sample(index)

                if (sample['r_window']['seq_len'] >= self.config['league']['window_min_size'] and 
                    sample['d_window']['seq_len'] >= self.config['league']['window_min_size']):

                    self.cache[cached_num] = sample
                    cached_num+=1

        self.__generated = True
        return self


    def __len__(self):
        return len(self.cache)


    def __getitem__(self, idx) -> dict:
        assert self.__generated
        return self.cache[idx]


    def generate_sample(self, index, corpus=None) -> dict:
        output = {}

        corpus = self.corpus if corpus is None else corpus

        # Previous games
        games = corpus[:index]
        # Current game
        anchor = corpus[index:index+1]

        r_team_id = anchor['r_team_id'].values.astype('int64')[0]
        d_team_id = anchor['d_team_id'].values.astype('int64')[0]

        r_players = anchor[[f'{s}_account_id' for s in self.RADIANT_SIDE]].values.astype('int64')[0]
        d_players = anchor[[f'{s}_account_id' for s in self.DIRE_SIDE]].values.astype('int64')[0]

        output['r_window'] = self.collector.collect_window(
            games=games, 
            team_id=r_team_id, 
            players=r_players)

        output['d_window'] = self.collector.collect_window(
            games=games, 
            team_id=d_team_id, 
            players=d_players)

        output['match_id'] = anchor['match_id'].values.astype('int64')

        match self.y_output:
            case 'binary':
                output['y'] = anchor['radiant_win'].values.astype('float32')
            case 'crossentropy':
                output['y'] = anchor['radiant_win'].values.astype('int64') + 1
        
        tabular_features = self.config['league']['features']['tabular']
        for feature in tabular_features:
            if tabular_features[feature]:
                match feature:
                    case 'teams':
                        output['teams'] = {
                            'radiant': r_team_id,
                            'dire': d_team_id,
                        }
                    case 'players':
                        output['players'] = {
                            'radiant': r_players,
                            'dire': d_players,
                        }
        
        return output


