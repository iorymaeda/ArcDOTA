from typing import TypedDict as D

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