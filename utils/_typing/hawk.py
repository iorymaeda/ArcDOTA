from typing import Literal, TypedDict as D

dota_match_id = int
hawk_team_id = int
hawk_match_id = int
hawk_league_id = int

class _hawk_hero(D):
    name: str
    code_name: str
    is_radiant: bool

class _odd(D):
    r_odd: float
    d_odd: float
    # -3 hours (-10800 in seconds) if converted to my timezone
    # -6 hours (-21600 in seconds) if converted to GMT
    live: bool
    created_at: int

class _match_team(D):
    id: hawk_team_id
    name: str
    logo_url: str

class Match(D):
    """Output by wrapper"""
    id: hawk_match_id
    match_id: list[dota_match_id]
    championship_name: str | None
    is_radiant_won: bool | None
    radiant: _match_team
    dire: _match_team
    radiant_heroes: list[str, str, str, str, str]
    dire_heroes: list[str, str, str, str, str]
    odds: dict[str, list[_odd]] | None

class _series_team(D):
    name: str
    logo_url: str

class _series_match(D):
    id: hawk_match_id
    number: int
    is_team1_radiant: bool
    is_radiant_won: bool | None
    heroes: list[_hawk_hero]

class Series(D):
    id: hawk_league_id
    team1: _series_team
    team2: _series_team
    best_of: Literal[1, 2, 3, 5]
    matches: list[_series_match]



