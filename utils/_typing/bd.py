from typing import TypedDict as D


class __SENDED_PREDICTIONS(D):
    match_id: int
    message_id: int
    prematch_time: int
    live_time: int
    is_prematch: bool
    is_live: bool

class Discord(D):
    SENDED_PREDICTIONS: __SENDED_PREDICTIONS


class __SCANED_GAMES(D):
    match_id: int
    radiant_win: bool
    pred_prematch: float
    pred_drafts: float

    radiant_name: str
    radiant_team_id: int
    radiant_series_win: int

    dire_name: str
    dire_team_id: int
    dire_series_win: int
    
    model_teg: str
    league_id: int
    founded_time: int

class Predictions(D):
    SCANED_GAMES: __SCANED_GAMES
    