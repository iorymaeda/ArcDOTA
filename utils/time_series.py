"""This module for any type of time-series"""

import numpy as np
import pandas as pd 
from typing import Literal

from . import _typing
from .tokenizer import Tokenizer
from .base import ConfigBase


class TSCollector(ConfigBase):
    def __init__(self, mask_type: str='bool', y_output: Literal['binary', 'crossentropy']='binary', teams_reg_output:bool=True):
        assert y_output in ['binary', 'crossentropy']

        self.y_output = y_output
        self.mask_type = mask_type
        self.teams_reg_output = teams_reg_output
        self.output_framework = np.array #//TODO
        self.config = self._get_config('features')
        #//TODO:fix it
        self.tokenizer = Tokenizer(path=self._get_curernt_path() / '../parse/output/tokenizer_league.pkl')
        

    def collect_window(
        self, games: pd.DataFrame,
        team_id: int, players: list | np.ndarray,
        mask_type:str=None, tokenize:bool=True,
        ) -> dict:
        """Collect previous team's games

        Here are a few rules for the search defined by configs.
        Args:
            - games: pd.DataFrame        - previous DOTA/team games in ascending chronological order
            - team_id: int               - team id in in for which we will collect window
            - players: list | np.ndarray - current players stack

        Returns:
            - output: dict - dict with windowed arrays with keys(features) defined in configs
                also returns `padded_mask`: np.array - padded mask for transformers/RNNs
                also returns `seq_len`: int - that determinate how many games in window
        """
        
        output = {}
        if isinstance(games, pd.DataFrame):
            games_bool, sides_bool = self.find_games(team_id, games, players)
            
            window = self.config['league']['window_size']
            cropped = games[games_bool][-window:]
            sides_bool = sides_bool[-window:]

            if tokenize and len(cropped):
                cropped = self.tokenizer.tokenize(cropped, players=True, teams=True)

            output['seq_len'] = len(cropped)
            output['padded_mask'] = self.generate_padding_mask(
                target_size=window,
                seq_len=len(cropped),
                mask_type=mask_type if mask_type else self.mask_type,
            )

            window_features: dict = self.config['league']['features']['window']
            for feature in window_features:
                # Can be False or None, so that means we need skip this one
                if window_features[feature]:
                    match feature:
                        case 'stats':
                            output[feature] = self.__parse_stats(cropped, sides_bool, window)

                        case 'opponent_stats':
                            output[feature] = self.__parse_stats(cropped, sides_bool, window, opponent=True)

                        case 'opponent':
                            output[feature] = self.__parse_opponents(cropped, sides_bool, window)

                        case 'result':
                            output[feature] = self.__parse_result(cropped, sides_bool, window)

                        case _:
                            raise Exception(feature)
                        
            
            return output

        else: raise NotImplementedError
        


    def collect_windows(
        self, games: pd.DataFrame, anchor: pd.DataFrame,
        mask_type:str=None, tokenize:bool=True, 
        ) -> dict:

        output = {}

        r_team_id = anchor['r_team_id'].values.astype('int64')[0]
        d_team_id = anchor['d_team_id'].values.astype('int64')[0]

        r_players = anchor[[f'{s}_account_id' for s in self.RADIANT_SIDE]].values.astype('int64')[0]
        d_players = anchor[[f'{s}_account_id' for s in self.DIRE_SIDE]].values.astype('int64')[0]

        output['match_id'] = anchor['match_id'].values.astype('int64')
        output['r_window'] = self.collect_window(
            games=games, team_id=r_team_id, players=r_players,
            mask_type=mask_type, tokenize=tokenize
        )
        output['d_window'] = self.collect_window(
            games=games, team_id=d_team_id, players=d_players,
            mask_type=mask_type, tokenize=tokenize
        )

        match self.y_output:
            case 'binary':
                output['y'] = anchor['radiant_win'].values.astype('float32')
            case 'crossentropy':
                output['y'] = anchor['radiant_win'].values.astype('int64')[0]
        
        if self.teams_reg_output:
            output['y_r_stats'] = anchor[[f"r_{f}" for f in _typing.property.FEATURES]].values.astype('float32')
            output['y_d_stats'] = anchor[[f"d_{f}" for f in _typing.property.FEATURES]].values.astype('float32')

        tabular_features = self.config['league']['features']['tabular']
        for feature in tabular_features:
            if tabular_features[feature]:
                match feature:
                    case 'teams':
                        output['teams'] = {
                            'radiant': self.tokenizer.tokenize(r_team_id, teams=True) if tokenize else r_team_id,
                            'dire': self.tokenizer.tokenize(d_team_id, teams=True) if tokenize else d_team_id,
                        }
                    case 'players':
                        output['players'] = {
                            'radiant': [self.tokenizer.tokenize(player, players=True) for player in r_players] if tokenize else r_players,
                            'dire': [self.tokenizer.tokenize(player, players=True) for player in d_players] if tokenize else d_players,
                        }
        return output


    @staticmethod
    def compare_players(key: np.ndarray, quary: np.ndarray, matching: int) -> bool:
        return True if len(set(key) & set(quary)) >= matching else False


    def generate_padding_mask(self, target_size:int, seq_len:int, mask_type: str) -> np.ndarray:
        """Generate a padding mask

        For a binary mask, a `True` value indicates that the corresponding key value will be IGNORED for the purpose of attention. 
        For a byte mask, a non-zero value indicates that the corresponding key value will be IGNORED.

        
        """
        assert target_size >= seq_len

        mask = np.ones(target_size)
        if target_size - seq_len > 0 and seq_len > 0:
            match self.config['league']['window_pad_mode']:
                case 'start':
                    mask[-seq_len:] = 0
                case 'end':
                    mask[:seq_len] = 0
                case _:
                    raise Exception

        return mask.astype(mask_type)
        

    def find_games(self, team_id: int, 
        games: pd.DataFrame, players: list | np.ndarray,
        ) -> list[np.ndarray, np.ndarray]:
        """Find correct games where this team play 

        returns:
            - games_bool: np.ndarray - bool array that determinate a good games
                len(games_bool) == len(games) 

            - sides_bool: np.ndarray - bool array that determinate team play as Radiant (True) or as Dire  (False)
                contains as many values as good games in `games_bool`
                len(sides_bool) <= len(games)
        """
        if isinstance(games, pd.DataFrame):
            # ------------------------------ #
            _bool: np.ndarray = games[['r_team_id', 'd_team_id']].values == team_id

            # Correct games
            games_bool: np.ndarray = _bool.sum(dtype=bool, axis=-1)
            # is this team play as radiant
            sides_bool = _bool[games_bool, 0]

            players_r = games[games_bool] \
                [[f'{s}_account_id' for s in self.RADIANT_SIDE]] \
                    .values.astype('int32')

            players_d = games[games_bool] \
                [[f'{s}_account_id' for s in self.DIRE_SIDE]] \
                    .values.astype('int32')

            # ------------------------------ #
            players_correct_games = []
            for side, r, d in zip(sides_bool, players_r, players_d):
                k_players = r if side else d
                players_correct_games.append(
                    self.compare_players(
                        key=k_players, 
                        quary=players, 
                        matching=self.config['league']['players_rules']['min_match_players_in_window']
                        )
                )

            # ------------------------------ #
            # update games_bool: drop uncorrect games
            games_bool[games_bool] = np.array(players_correct_games, dtype='bool')
            # update sides_bool: drop uncorrect games
            sides_bool = _bool[games_bool, 0]

            return games_bool, sides_bool

        else: raise NotImplementedError


    def __parse_result(self, cropped: pd.DataFrame, sides_bool: np.ndarray, window: int) \
        -> np.ndarray:
        """Parse result (win or lose) from previous games

        Args:
            - cropped: pd.DataFrame - dataframe with valid previous games
            - sides_bool: np.ndarray - bool array that determinate anchor team play 
                as Radiant (True) or as Dire  (False)
            - window: int - how many parse previous games 
            
        Output shape: (`window`, 1)
        """

        if isinstance(cropped, pd.DataFrame):
            output = []
            r_wins = cropped['radiant_win'].values

            for side, win in zip(sides_bool, r_wins):
                output.append(win if side else not win)

            if self.config['league']['features']['window']['result'] == 'categorical':
                dtype = 'int64'
            elif self.config['league']['features']['window']['result'] == 'linear':
                dtype = 'float32'
            else:
                raise NotImplementedError

            if len(output) == 0:
                shape = list(r_wins.shape)
                shape[0] = window
                output = np.zeros(shape, dtype=dtype)
            else:
                output = np.array(output, dtype=dtype)
                output = self.pad_window(output, window, self.config['league']['window_pad_mode'])

            return output
        else: raise NotImplementedError


    def __parse_stats(self, cropped: pd.DataFrame, sides_bool: np.ndarray, window: int, opponent=False) \
        -> np.ndarray:
        """Parse statistic from previous games

        Output shape: (`window`, F) 
        F is num of features from `_typing.property.FEATURES`
        """
        if isinstance(cropped, pd.DataFrame):
            output = []
            columns_r = [f"r_{f}" for f in _typing.property.FEATURES]
            columns_d = [f"d_{f}" for f in _typing.property.FEATURES]
            stats_r = cropped[columns_r].values.astype('float32')
            stats_d = cropped[columns_d].values.astype('float32')

            if opponent:
                for side, r_stat, d_stat in zip(sides_bool, stats_r, stats_d):
                    output.append(d_stat if side else r_stat)
            else:
                for side, r_stat, d_stat in zip(sides_bool, stats_r, stats_d):
                    output.append(r_stat if side else d_stat)

            if opponent:
                if self.config['league']['features']['window']['opponent_stats'] == 'categorical':
                    raise NotImplementedError
                elif self.config['league']['features']['window']['opponent_stats'] == 'linear':
                    dtype = 'float32'
                else:
                    raise Exception

            else:
                if self.config['league']['features']['window']['stats'] == 'categorical':
                    raise NotImplementedError
                elif self.config['league']['features']['window']['stats'] == 'linear':
                    dtype = 'float32'
                else:
                    raise Exception

            if len(output) == 0:
                shape = list(stats_r.shape)
                shape[0] = window
                output = np.zeros(shape, dtype=dtype)
            else:
                output = np.array(output, dtype=dtype)
                output = self.pad_window(output, window, self.config['league']['window_pad_mode'])

            return output
        else: raise NotImplementedError


    def __parse_opponents(self, cropped: pd.DataFrame, sides_bool: np.ndarray, window: int) \
        -> np.ndarray:
        """Parse anchor team's opponents from previous games

        Args:
            - cropped: pd.DataFrame - dataframe with valid previous games
            - sides_bool: np.ndarray - bool array that determinate anchor team play 
                as Radiant (True) or as Dire  (False)
            - window: int - how many parse previous games 
            
        Output shape: (`window`, 1)
        """

        if isinstance(cropped, pd.DataFrame):
            output = []
            teams_r = cropped['r_team_id'].values.astype('int64')
            teams_d = cropped['d_team_id'].values.astype('int64')

            for side, r_team, d_team in zip(sides_bool, teams_r, teams_d):
                output.append(d_team if side else r_team)

            if self.config['league']['features']['window']['opponent'] == 'categorical':
                dtype = 'int64'
            else:
                raise Exception

            if len(output) == 0:
                shape = list(teams_r.shape)
                shape[0] = window
                output = np.zeros(shape, dtype=dtype)
            else:
                output = np.array(output, dtype=dtype)
                output = self.pad_window(output, window, self.config['league']['window_pad_mode'])

            return output
        else: raise NotImplementedError


    def pad_window(self, arr: np.ndarray, window: int, mode:Literal['start', 'end']='end', constant:int=0):
        """Pad array to window size

        Args:
            - arr: np.ndarray - null
            - window: int - null
            - mode: str - null
            - constant: int - null

        Input shape: (Any, ...)
        Output shape: (`window`, ...)"""
        assert mode in ['end', 'start']

        arr = arr[-window:]
        l = len(arr)
        if l < window:
            shape = list(arr.shape)
            shape[0] = window

            padded_arr = np.ones(shape, dtype=arr.dtype)
            padded_arr*= constant

            if l > 0:
                if mode == 'start':
                    padded_arr[-l:] = arr
                elif mode == 'end':
                    padded_arr[:l] = arr
            return padded_arr

        return arr

