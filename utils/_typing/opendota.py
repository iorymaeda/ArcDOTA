from typing import List, Literal
from typing import TypedDict as D

match_id = int
leaguetiers = Literal["professional", "premium", "amateur", "excluded"]
# "amateur", "excluded" is trash:
#
# "leagueid": 12582,
# "ticket": null,
# "banner": null,
# "tier": "excluded",
# "name": "Кубанский союз молодёжи"

class Player(D):
    """Opendota player model"""
    player_slot: int
    total_gold: int
    total_xp: int
    kills: int
    deaths: int
    assists: int
    lh_t: list[int]
    dn_t: list[int]
    roshan_kills: int
    tower_damage: int
    hero_damage: int

class Objectives(list):
    pass

class Players(list):
    pass

class Match(D):
    """Opendota match"""
    objectives: list[dict]
    players: list[Player]

class TeamMatch(D):
    match_id: match_id
    radiant_win: bool
    radiant: bool
    duration: int
    start_time: int
    leagueid: int
    league_name: str
    cluster: int
    opposing_team_id: int
    opposing_team_name: str
    opposing_team_logo: str

TeamMatches = List[TeamMatch]

class SQLPublicMatch(D):
    match_id: match_id
    match_seq_num: int
    radiant_win: bool
    start_time: int
    duration: int
    avg_mmr: int
    num_mmr: int
    lobby_type: int
    game_mode: int
    avg_rank_tier: int
    num_rank_tier: int
    cluster: int

SQLPublicMatches = List[SQLPublicMatch]

class SQLLeagueMatch(D):
    match_id: match_id
    leaguename: str
    leaguetier: leaguetiers
    start_time: int
    lobby_type: int
    human_players: int
    game_mode: int
    radiant_team_id: int
    dire_team_id: int
    radiant_captain: int
    dire_captain: int

SQLLeagueMatches = List[SQLLeagueMatch]

class League(D):
    leagueid: int
    ticket: None | str
    banner: None | str
    tier: leaguetiers
    name: str

Leagues = List[League]

class TeamPlayers(D):
    account_id: int
    name: str
    games_played: int
    wins: int
    is_current_team_member: bool