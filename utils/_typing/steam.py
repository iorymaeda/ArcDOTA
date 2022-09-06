from typing import Literal, List
from typing import TypedDict as D


class _league_team(D):
    team_name: str
    team_id: int
    team_logo: int
    complete: bool

class _league_scoreboard_player(D):
    player_slot: int
    account_id: int
    hero_id: int
    kills: int
    death: int
    assists: int
    last_hits: int
    denies: int
    gold: int
    level: int
    gold_per_min: int
    xp_per_min: int
    ultimate_state: int
    ultimate_cooldown: int
    item0: int
    item1: int
    item2: int
    item3: int
    item4: int
    item5: int
    respawn_timer: int
    position_x: float
    position_y: float
    net_worth: int

class _league_team_scoreboards(D):
    score: int
    tower_state: int
    barracks_state: int
    picks: List[ dict[Literal['hero_id'], int] ]
    bans: List[ dict[Literal['hero_id'], int] ]
    players: List[_league_scoreboard_player]
    abilities: list[ dict[Literal['ability_id'], int], dict[Literal['ability_level'], int] ]

class _league_scorebord(D):
    duration: float
    roshan_respawn_timer: int
    radiant: _league_team_scoreboards
    dire: _league_team_scoreboards

class _league_game_player(D):
    account_id: int
    name: str
    hero_id: int
    team: int

class _league_game_base(D):
    players: List[_league_game_player]
    lobby_id: int
    match_id: int
    spectators: int
    league_id: int
    league_node_id: int
    stream_delay_s: int
    radiant_series_wins: int
    dire_series_wins: int
    series_type: int
    scorebord: _league_scorebord

class _league_game(_league_game_base, total=False):
    radiant_team: _league_team
    dire_team: _league_team

class _GetLiveLeagueGames_result(D):
    games: List[_league_game]
    status: int

class GetLiveLeagueGames(D):
    '/IDOTA2Match_570/GetLiveLeagueGames/v1?key={api_key}'
    result: _GetLiveLeagueGames_result

class _GetTournamentPrizePool_result(D):
    prize_pool: int
    league_id: int
    status: int

class GetTournamentPrizePool(D):
    '/IEconDOTA2_570/GetTournamentPrizePool/v1?key={api_key}&{leagueid=}'
    result: _GetTournamentPrizePool_result
