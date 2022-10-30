import pydantic
from pydantic import BaseModel as B
from pydantic.dataclasses import dataclass as D

# Use this to parse json to pydantic model
# pydantic.parse_obj_as(utils._typing.property.Match, league[0])
# utils._typing.property.Match.parse_obj(league[0])

class Stats(B):
    gold: int = 0 
    xp: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    last_hits: int = 0
    last_hits_10: int = 0
    denies_5: int = 0
    denies_10: int = 0
    roshan_kills: int = 0
    tower_damage: int = 0
    attack_damage: int = 0
    spell_damage: int = 0
    stuns: float = 0
    heal: int = 0

    # ------------------- #
    # In my opinion this features not rly depended on heroes
    obs_placed: int = 0
    sen_placed: int = 0
    camps_stacked: int = 0
    creeps_stacked: int = 0
    
class TimeSeries(B):
    gold: list[int]
    kills: list[int]
    xp: list[int]
    lh: list[int]
    dn: list[int]
    

class Team(B):
    id: int


class Teams(B):
    radiant: Team
    dire: Team


class Player(B):
    """leaver_status
    0 - NONE - finished match, no abandon.
    1 - DISCONNECTED - player DC, no abandon.
    2 - DISCONNECTED_TOO_LONG - player DC > 5min, abandoned.
    3 - ABANDONED - player DC, clicked leave, abandoned.
    4 - AFK - player AFK, abandoned.
    5 - NEVER_CONNECTED - player never connected, no abandon.
    6 - NEVER_CONNECTED_TOO_LONG - player took too long to connect, no abandon."""
    slot: int
    hero_id: int
    # top n by total gold from 1 to 5
    hero_role: int 
    lane_role: int
    rank_tier: int | None
    account_id: int | None
    leaver_status: int


class Players(B):
    radiant: list[Player]
    dire: list[Player]


class League(B):
    id: int
    name: str
    # tier: Any - 'professional' | 'premium' | 'excluded'
    # premium is official valve tournaments
    # professional is unofficial tournaments
    # excluded is smth like trash
    tier: str
    prize_pool: int
    

class PlayerOverview(B):
    slot: int
    stats: Stats
    time_series: TimeSeries | None


class TeamOverview(B):
    stats: Stats
    time_series: TimeSeries | None


class Overview(B):
    players: list[PlayerOverview]
    teams: list[TeamOverview]


class Match(B):
    match_id: int
    start_time: int
    lobby_type: int
    game_mode: int
    version: int
    region: int | None

    duration: int
    radiant_win: bool

    isleague: bool
    league: League | None
    series_type: int
    series_id: int

    players: Players
    teams: Teams | None

    overview: Overview


__schema: dict = pydantic.schema_of(Stats)
__schema: dict = __schema['definitions']['Stats']['properties']
FEATURES: list = list(__schema.keys())