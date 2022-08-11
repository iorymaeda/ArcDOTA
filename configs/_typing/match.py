from typing import TypedDict as D


class rd(D):
    """Summary `r_` + `d_` stats"""
    min: dict
    max: dict
    
class r_d(D):
    """`r_` or `d_` stats"""
    min: dict
    max: dict
    
class match(D):
    """Config for drop matches"""
    min: dict
    max: dict
    rd: rd
    r_d: r_d