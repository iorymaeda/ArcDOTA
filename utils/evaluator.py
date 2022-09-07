"""This class drops bad matches / evaluate as good or bad a match"""
import logging
from contextlib import suppress

import pandas as pd

from . import _typing 
from .base import ConfigBase


class Evaluator(ConfigBase):
    """Evaluate match as good or bad
    bad matches will be dropped
    good/bad matches are defined by configs: `../configs/match.yaml`"""
    def __init__(self):
        self.config = self._get_config('match')


    def __call__(self, match: _typing.property.Match | dict | pd.DataFrame) -> pd.DataFrame | bool:
        if isinstance(match, _typing.property.Match):
            if match.isleague:
                if self.evaluate_league(match):
                    return self.evaluate_match(match)

                else: return False

        elif isinstance(match, dict):
            if match['isleague']:
                if self.evaluate_league(match):
                    return self.evaluate_match(match)

                else: return False

        elif isinstance(match, pd.DataFrame):
            if match['isleague'].sum() > 0:
                match = self.evaluate_league(match)

            return self.evaluate_match(match)

        else:
            raise Exception

    def evaluate_league(self, match: _typing.property.Match | dict | pd.DataFrame) \
        -> pd.DataFrame | bool:
        """Evaluate league match/matches
        if one match is passed (`typing.property.Match` | `dict`)
            returns bool 
        if few matches is passed (`pd.DataFrame`) 
            returns only good matches or empty Dataframe if there no one good game
        """
        config = self.config['league']

        if isinstance(match, pd.DataFrame):
            if len(match) == 0:
                return match

            bad_matches = pd.Series(
                data=[False for _ in range(len(match))], 
                index=match.index,
            )
            excepted_tiers = bad_matches.copy()
            league_bool = match['isleague']
            # ------------------------------ #
            for tier in config['tiers']:
                bool_ = match['league_tier'] == tier
                with suppress(TypeError, KeyError):
                    if config['tiers'][tier]['include']:
                        b = match['league_prize_pool'][bool_] < config['tiers'][tier]['prize_pool']
                        bad_matches[bool_] += b
                        logging.info(f'{tier} league_prize_pool: {b.sum()}')
                    else:
                        bad_matches += bool_

                excepted_tiers += bool_
 
            # process unexcepted tier
            unexcepted_tiers = ~excepted_tiers
            if config['tiers']['others']['include']: 
                b = match['league_prize_pool'][unexcepted_tiers] < config['tiers']['others']['prize_pool']
                bad_matches[unexcepted_tiers] += b
                logging.info(f'include league_prize_pool: {b.sum()}')
            else:
                bad_matches += unexcepted_tiers
            
            bad_matches = bad_matches.astype('bool')
            # public matches should be `good`
            bad_matches[~league_bool] = False

            # Return good matches
            return match[~bad_matches]

        
        elif isinstance(match, _typing.property.Match):
            with suppress(TypeError, KeyError):
                for tier in config['tiers']:
                    if (match.league):
                        if match.league.tier == tier:
                            if config[tier]['include']:
                                return match.league.prize_pool >= config['tiers'][tier]['prize_pool']
                            else:
                                return False
                    else:
                        return False

                
            # process unexcepted tier
            if config['tiers']['others']['include']:
                if (match.league.prize_pool):
                    return match.league.prize_pool >= config['tiers']['others']['prize_pool']
                else:
                    return False
            else:
                return False


        elif isinstance(match, dict):
            with suppress(TypeError, KeyError):
                for tier in config['tiers']:
                    if match['league']['tier'] == tier:
                        if config[tier]['include']:
                            return match['league']['prize_pool'] >= config['tiers'][tier]['prize_pool']
                        else:
                            return False
                
            # process unexcepted tier
            if config['tiers']['others']['include']:
                return match['league']['prize_pool'] >= config['tiers']['others']['prize_pool']
            else:
                return False

        else:
            raise Exception


    def evaluate_match(self, match: _typing.property.Match | dict | pd.DataFrame) \
        -> pd.DataFrame | bool:
        """Evaluate match/matches
        if one match is passed (`typing.property.Match` | `dict`)
            returns bool 
        if few matches is passed (`pd.DataFrame`) 
            returns only good matches or empty Dataframe if there no one good game
        """

        config = self.config['public']

        if isinstance(match, pd.DataFrame):
            if len(match) == 0:
                return match

            bad_matches = pd.Series(
                data=[False for _ in range(len(match))], 
                index=match.index,
            )
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['min']:
                    b = match[f] < config['min'][f]
                    bad_matches += b
                    logging.info(f'min {f}: {b.sum()}')

            with suppress(TypeError, KeyError):
                for f in config['max']:
                    b = match[f] > config['max'][f]
                    bad_matches += b
                    logging.info(f'max {f}: {b.sum()}')
                    
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['rd']['min']:
                    b = (match[f"r_{f}"] + match[f"d_{f}"]) < config['rd']['min'][f]
                    bad_matches += b
                    logging.info(f'rd min {f}: {b.sum()}')

            with suppress(TypeError, KeyError):
                for f in config['rd']['max']:
                    b = (match[f"r_{f}"] + match[f"d_{f}"])> config['rd']['max'][f]
                    bad_matches += b
                    logging.info(f'rd max {f}: {b.sum()}')

            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['r_d']['min']:
                    b = (match[f"r_{f}"] < config['r_d']['min'][f]) | (match[f"d_{f}"] < config['r_d']['min'][f])
                    bad_matches += b
                    logging.info(f'r_d min {f}: {b.sum()}')
            
            with suppress(TypeError, KeyError):
                for f in config['r_d']['max']:
                    b = (match[f"r_{f}"] > config['r_d']['max'][f]) | (match[f"d_{f}"] > config['r_d']['max'][f])
                    bad_matches += b
                    logging.info(f'r_d max {f}: {b.sum()}')

            # Return good matches
            return match[~bad_matches]


        elif isinstance(match, dict):
            r = match['overview']['teams'][0]
            d = match['overview']['teams'][1]
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['min']:
                    if match[f] < config['min'][f]:
                        return False

            with suppress(TypeError, KeyError):
                for f in config['max']:
                    if match[f] > config['max'][f]:
                        return False

            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['rd']['min']:
                    if r[f] + d[f] < config['rd']['min'][f]:
                        return False

            with suppress(TypeError, KeyError):
                for f in config['rd']['max']:
                    if r[f] + d[f] > config['rd']['max'][f]:
                        return False
            
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['r_d']['min']:
                    if (r[f] < config['r_d']['min'][f]) or (d[f] < config['r_d']['min'][f]):
                        return False
            
            with suppress(TypeError, KeyError):
                for f in config['r_d']['max']:
                    if (r[f] < config['r_d']['max'][f]) or (d[f] < config['r_d']['max'][f]):
                        return False

            return True

        elif isinstance(match, _typing.property.Match):
            teams_overview = getattr(getattr(match, 'overview'), 'teams')
            r = getattr(teams_overview[0], 'stats')
            d = getattr(teams_overview[1], 'stats')
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['min']:
                    if getattr(match, f) < config['min'][f]:
                        return False

            with suppress(TypeError, KeyError):
                for f in config['max']:
                    if getattr(match, f) > config['max'][f]:
                        return False

            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['rd']['min']:
                    if getattr(r, f) + getattr(d, f) < config['rd']['min'][f]:
                        return False

            with suppress(TypeError, KeyError):
                for f in config['rd']['max']:
                    if getattr(r, f) + getattr(d, f) > config['rd']['max'][f]:
                        return False
            
            # ------------------------------ #
            with suppress(TypeError, KeyError):
                for f in config['r_d']['min']:
                    if (getattr(r, f) < config['r_d']['min'][f]) or (getattr(d, f) < config['r_d']['min'][f]):
                        return False
            
            with suppress(TypeError, KeyError):
                for f in config['r_d']['max']:
                    if (getattr(r, f) < config['r_d']['max'][f]) or (getattr(d, f) < config['r_d']['max'][f]):
                        return False

            return True

        else:
            raise Exception

