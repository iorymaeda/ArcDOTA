import numpy as np
import pandas as pd

from .base import ConfigBase


class Tokenizer(ConfigBase):
    RADIANT_SIDE = [0, 1, 2, 3, 4]
    DIRE_SIDE = [128, 129, 130, 131, 132]
    def __init__(self):
        self.__fitted = False

        self.config = self.get_config('features.yaml')
        
        self.players_vocab = {}
        self.players_entering = {}
        self.player_converter = None

        self.teams_vocab = {}
        self.teams_entering = {}
        self.team_converter = None


    @property
    def fitted(self):
        return self.__fitted


    def fit(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self.teams_entering = self.__fit_teams(df)
            self.players_entering = self.__fit_players(df)

            self.teams_vocab = self.__build_teams_vocab()
            self.players_vocab = self.__build_players_vocab()

            self.__build_vectorizers()

            self.__fitted = True
        else: raise NotImplementedError


    def __build_vectorizers(self):
        self.team_converter = np.vectorize(self.__converter(self.teams_vocab))
        self.player_converter = np.vectorize(self.__converter(self.players_vocab))


    def __converter(self, vocab: dict):
        def converter(id: np.int16 | np.int32 | np.int64) -> int:
            if id == 0: return 0

            token = vocab.get(id)
            if token: return token
            else: return 1 

        return converter


    def tokenize(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()
        if self.__fitted:
            _ = [f'{s}_account_id' for s in self.RADIANT_SIDE + self.DIRE_SIDE]
            players_arr = _df[_].astype('int64').values
            teams_arr = _df[['r_team_id', 'd_team_id']].astype('int64').values

            _df[_] = self.player_converter(players_arr)
            _df[['r_team_id', 'd_team_id']] = self.team_converter(teams_arr)

            return _df 
        else: raise Exception('Not fitted')


    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()
        if self.__fitted:
            _ = [f'{s}_account_id' for s in self.RADIANT_SIDE + self.DIRE_SIDE]
            b = ((5 - (_df[_].values[:,:5] == 1).sum(axis=1) >= self.config['league']['players_rules']['min_good_players_in_stack']) &\
            (5 - (_df[_].values[:,5:] == 1).sum(axis=1) >= self.config['league']['players_rules']['min_good_players_in_stack']))

            _df = _df[b]
            _df = _df[(_df[['r_team_id', 'd_team_id']] == 1).sum(axis=1) == 0]

            return _df 
        else: raise Exception('Not fitted')


    def __build_players_vocab(self, const=None) -> dict[int, int]:
        if const is None:
            const = self.config['league']['players_rules']['min_played_games']

        players_key_value_arr = np.array([[k, v] for k, v in self.players_entering.items()])
        good_players = players_key_value_arr[players_key_value_arr[:, 1] > const]

        players_tokens = {}
        for idx, k in enumerate(good_players[:, 0]):
            players_tokens[int(k)] = idx + 2

        return players_tokens


    def __build_teams_vocab(self, const=None) -> dict[int, int]:
        if const is None:
            const = self.config['league']['teams_rules']['min_played_games']

        teams_key_value_arr = np.array([[k, v] for k, v in self.teams_entering.items()])
        good_teams = teams_key_value_arr[teams_key_value_arr[:, 1] > const]

        teams_tokens = {}
        for idx, k in enumerate(good_teams[:, 0]):
            teams_tokens[int(k)] = idx + 2

        return teams_tokens


    def __fit_players(self, df: pd.DataFrame) -> dict[int, int]:
        players_arr = df[[f'{s}_account_id' for s in self.RADIANT_SIDE + self.DIRE_SIDE]].astype('int64').values
        flatten_players_arr = players_arr.flatten()
        vocab = set(flatten_players_arr)

        played_games = {p:0 for p in vocab}
        for p in flatten_players_arr:
            played_games[p] += 1
            
        try: played_games[0] = 0
        except: pass
            
        return played_games
        

    def __fit_teams(self, df: pd.DataFrame) -> dict[int, int]:
        teams_arr = df[['r_team_id', 'd_team_id']].astype('int64').values
        flatten_teams_arr = teams_arr.flatten()
        vocab = set(flatten_teams_arr)

        played_games = {p:0 for p in vocab}
        for p in flatten_teams_arr:
            played_games[p] += 1
            
        try: played_games[0] = 0
        except: pass

        return played_games

        