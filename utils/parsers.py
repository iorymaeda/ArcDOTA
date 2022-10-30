import json
import pathlib

import pydantic
import numpy as np
import pandas as pd

from . import _typing, exceptions
from .base import DotaconstantsBase
from .development import suppress


class OpendotaParser(DotaconstantsBase):
    """Parse opendota match to property standart"""

    def __init__(self, tokenizer:str|dict=None):
        self.patch: dict
        self.region: dict
        self.hero_json: dict
        self.npc_to_id: dict
        self.id_to_hero: dict
        self.id_to_npc: dict
        self.lobby_type: dict
        self.game_mode: dict
        self.leagues: dict
        self.prize_pools: dict
        
        # tokenizer for events, literraly not implemented
        if isinstance(tokenizer, str):
            raise NotImplementedError
            self.tokenizer = self._load_tokenizer(path=tokenizer)
        elif isinstance(tokenizer, dict):
            raise NotImplementedError
            self.tokenizer = tokenizer
        elif tokenizer is None:
            tokenizer = {}
        else:
            raise Exception("`tokenizer` type must be str or dict")

        dotaconstants_path = self._get_constants_path()
        patch_path = f'{dotaconstants_path}/build/patch.json'
        region_path = f'{dotaconstants_path}/build/region.json' 
        game_mode_path = f'{dotaconstants_path}/build/game_mode.json'
        hero_names_path = f'{dotaconstants_path}/build/hero_names.json'
        lobby_type_path = f'{dotaconstants_path}/build/lobby_type.json'
        self._load_patch(patch_path)
        self._load_region(region_path)
        self._load_game_mode(game_mode_path)
        self._load_hero_names(hero_names_path)
        self._load_lobby_type(lobby_type_path)
        

        path = self._get_relative_path()
        path = path.parent.resolve()
        leagues_path = path / 'scarpe/output/leagues.json'
        prize_pools_path = path / 'scarpe/output/prize_pools.json'
        self._load_leagues(leagues_path)
        self._load_prize_pools(prize_pools_path)


    def _load_json(self, path: str | pathlib.Path) -> dict:
        for encoding in [None, 'utf8', 'utf16', 'utf32']:
            with (suppress(UnicodeDecodeError), open(path, 'r', encoding=encoding) as f):
                    return json.load(f)

        raise UnicodeDecodeError

    def _load_leagues(self, path: str):
        leagues = self._load_json(path)
        self.leagues = {
            l['leagueid']: {
                'tier': l['tier'],
                'name': l['name']
            } for l in leagues
        }

    def _load_prize_pools(self, path: str):
        prize_pools = self._load_json(path)
        self.prize_pools = {int(k):v for k, v in prize_pools.items()}

    def _load_lobby_type(self, path: str):
        self.lobby_type = self._load_json(path)

    def _load_game_mode(self, path: str):
        self.game_mode = self._load_json(path)

    def _load_region(self, path):
        self.region = self._load_json(path)

    def _load_patch(self, path):
        self.patch = self._load_json(path)

    def _load_hero_names(self, path):
        self.hero_json = self._load_json(path)
        self.npc_to_id, self.id_to_hero = self._load_heroes()
        self.id_to_npc = {self.hero_json[npc]['id']: npc for npc in self.hero_json}

    def _load_tokenizer(self, path):
        raise NotImplementedError
        
    def parse_hero_roles(self, players: _typing.opendota.Players) -> dict[int, int]:
        hero_role = {}
        Rgold_hero = {}
        Dgold_hero = {}
        for player in players:
            if player['player_slot'] in self.RADIANT_SIDE:
                Rgold_hero[player['hero_id']] = player['total_gold']
            elif player['player_slot'] in self.DIRE_SIDE:
                Dgold_hero[player['hero_id']] = player['total_gold']

        glist = {k: v for k, v in sorted(Rgold_hero.items(), key=lambda item: item[1], reverse=True)}
        glist = list(glist.keys())
        for idx, rg in enumerate(glist):
            hero_role[rg] = idx

        glist = {k: v for k, v in sorted(Dgold_hero.items(), key=lambda item: item[1], reverse=True)}
        glist = list(glist.keys())
        for idx, rg in enumerate(glist):
            hero_role[rg] = idx

        return hero_role


    def parse_building_destroys(self, objectives: _typing.opendota.Objectives) -> tuple[dict[int, int], int, int]:
        d = {s:0 for s in self.RADIANT_SIDE + self.DIRE_SIDE}
        r_twd_t = 0
        d_twd_t = 0
        for obj in objectives:
            if obj['type'] == 'building_kill':
                # Examples of `obj['key']`:
                # // npc_dota_goodguys_... - dire's tower
                # // npc_dota_badguys_...  - radiant's tower
                if obj['key'][9] == 'b':
                    r_twd_t+=1

                elif obj['key'][9] == 'g':
                    d_twd_t+=1    

                else: continue

                if 'player_slot' in obj:
                    d[int(obj['player_slot'])] += 1

        return d, r_twd_t, d_twd_t


    def parse_kills(self, players: _typing.opendota.Players, max_duration:int=63, max_kills:int=10):
        """
        : param players:            opendota match players
        : param max_duration:       match max duration in minutes, 
                                    this means that all data after this time point will be ignored
        """
        def parse_player(kills: np.ndarray):
            for f in player['kills_log']:
                if f['time'] < 0:
                    f['time'] = 0

                m = f['time']//60
                if m > max_duration: 
                    break
                
                if kills[m] < max_kills:
                    kills[m] += 1
                
            return kills
                            

        kills_r = np.zeros((max_duration+1), dtype=np.int32)
        kills_d = np.zeros((max_duration+1), dtype=np.int32)
        
        for player in players:
            if player['player_slot'] in self.RADIANT_SIDE:
                kills_r = parse_player(kills_r)
                
            if player['player_slot'] in self.DIRE_SIDE:
                kills_d = parse_player(kills_d)
            
        return kills_r, kills_d


    def parse_t(self, players: _typing.opendota.Players, t: str, max_duration:int=63) -> list[np.ndarray, np.ndarray]:
        """parse t in opendota match players, where n is gold, xp, e.t.c."""
        r = np.zeros((max_duration+1), dtype=np.int32)
        d = np.zeros((max_duration+1), dtype=np.int32)
        
        for player in players['players']:
            arr = np.array(player[t])
            arr = arr[:max_duration+1]
            for n_i in range(arr.shape[0] - 1):
                arr[n_i+1:] = arr[n_i+1:] - arr[n_i]

            padded_arr = np.zeros((max_duration+1), dtype=np.int32)
            padded_arr[:arr.shape[0]] = arr
            
            if player['player_slot'] in self.RADIANT_SIDE:
                r += padded_arr
            
            if player['player_slot'] in self.DIRE_SIDE:
                d += padded_arr
                
        return r, d


    def parse_events(self, objectives: _typing.opendota.Objectives, max_duration:int=63, log_pad_window:int=8):
        '''
        : param objectives:         opendota objectives
        : param max_duration:       match max duration in minutes, 
                                    this means that all data after this time point will be ignored
        : param log_pad_window:     number of values to storage events, in one minute may be a lot 
                                    of events, for example in 35 minute Radiant team: kill roshan,
                                    pick  aegis  and  destroy  2 towers and 2 barracks, while Dire 
                                    only destroy 2 towers
        '''
        raise NotImplementedError

        events = np.zeros((max_duration+1, log_pad_window), dtype=np.int32)

        for f in objectives:
            
            if f['time'] < 0:
                f['time'] = 0
                
            m = f['time']//60
            if m > max_duration: 
                break
                
            if f['type'] == 'building_kill':
                for idx, _ in enumerate(events[m]):
                    if _ == 0:
                        key = f['key']
                        if key in self.tokenizer:
                            events[m, idx] = self.tokenizer[key]
                        else:
                            self.tokenizer[key] = len(self.tokenizer) + 1
                        break

            elif f['type'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                for idx, _ in enumerate(events[m]):
                    if _ == 0:
                        key = 'radiant_kill_roshan' if f['team'] == 2 else 'dire_kill_roshan'
                        if key in self.tokenizer:
                            events[m, idx] = self.tokenizer[key]
                        else:
                            self.tokenizer[key] = len(self.tokenizer) + 1
                        break

            elif f['type'] == 'CHAT_MESSAGE_AEGIS':
                for idx, _ in enumerate(events[m]):
                    if _ == 0:
                        key = 'radiant_get_aegis' if f['player_slot'] in self.RADIANT_SIDE else 'dire_get_aegis'
                        if key in self.tokenizer:
                            events[m, idx] = self.tokenizer[key]
                        else:
                            self.tokenizer[key] = len(self.tokenizer) + 1
                        break
                        
        return events


    def __call__(self, match: _typing.opendota.Match) -> _typing.property.Match:
        """Convert opendota match to property match"""
        return _typing.property.Match(
            league=self.get_league(match) if ('leagueid' in match and match['leagueid'] > 0) else None,
            isleague=True if ('leagueid' in match and match['leagueid'] > 0) else False,
            series_type=match['series_type'],
            series_id=match['series_id'],

            match_id=match['match_id'],
            start_time=match['start_time'],
            lobby_type=match['lobby_type'],
            game_mode=match['game_mode'],
            version=match['version'],
            region=match['region'] if 'region' in match else None,

            duration=match['duration'],
            radiant_win=match['radiant_win'],

            players=self.get_players(match['players']),
            teams=self.get_teams(match),
            
            overview=self.get_overview(match),
        )


    def get_teams(self, match: _typing.opendota.Match) -> _typing.property.Teams | None:
        try:
            if 'radiant_team_id' in match and 'dire_team_id' in match:
                return _typing.property.Teams(
                    radiant=_typing.property.Team(id=match['radiant_team_id']),
                    dire=_typing.property.Team(id=match['dire_team_id']),
                )
            elif 'radiant_team' in match and 'dire_team' in match:
                return _typing.property.Teams(
                    radiant=_typing.property.Team(id=match['radiant_team']['team_id']),
                    dire=_typing.property.Team(id=match['dire_team']['team_id']),
                )
            else:
                return None

        except pydantic.ValidationError:
            return None


    def get_players(self, players: _typing.opendota.Players) -> _typing.property.Players:
        def _fill(player: _typing.opendota.Player) -> _typing.property.Player:
            return _typing.property.Player(
                    slot=player["player_slot"],
                    hero_id=player["hero_id"],
                    hero_role=roles[player["hero_id"]],
                    lane_role=player['lane_role'],
                    rank_tier=player['rank_tier'],
                    account_id=player['account_id'],
                    leaver_status=player['leaver_status'],
                )
        r, d = [], []
        roles = self.parse_hero_roles(players)
        for player in players:
            if player["player_slot"] in self.RADIANT_SIDE:
                _player = _fill(player)
                r.append(_player)

            elif player["player_slot"] in self.DIRE_SIDE:
                _player = _fill(player)
                d.append(_player)

        return _typing.property.Players(radiant=r, dire=d)
        

    def get_league(self, match: _typing.opendota.Match) -> _typing.property.League | None:
        try:
            leagueid = match['leagueid'] 
            if (leagueid not in self.leagues and 
                leagueid not in self.prize_pools): 
                raise exceptions.property.LeaguesJSONsNotFound(leagueid)

            elif leagueid not in self.leagues:
                raise exceptions.property.LeagueIDNotFound(leagueid)
            
            elif leagueid not in self.prize_pools:
                raise exceptions.property.LeaguePPNotFound(leagueid)

            return _typing.property.League(
                id=match['leagueid'],
                name=self.leagues[match['leagueid']]['name'],
                tier=self.leagues[match['leagueid']]['tier'],
                prize_pool=self.prize_pools[match['leagueid']],
            )
        except KeyError:
            return None


    def get_overview(self, match: _typing.opendota.Match) -> _typing.property.Overview:
        overview = _typing.property.Overview(players=[], teams=[])
        radiant_team_stats = _typing.property.Stats()
        dire_team_stats = _typing.property.Stats()

        
        radiant_npc = [ self.id_to_npc[ p['hero_id'] ] for p in match['players'] if p['player_slot'] in self.RADIANT_SIDE]
        dire_npc = [ self.id_to_npc[ p['hero_id'] ] for p in match['players'] if p['player_slot'] in self.DIRE_SIDE]

        for player in match['players']:
            player_stats = _typing.property.Stats()
            if player["player_slot"] in self.RADIANT_SIDE:
                obj = radiant_team_stats
                oppositing_npc = dire_npc

            elif player["player_slot"] in self.DIRE_SIDE:
                obj = dire_team_stats
                oppositing_npc = radiant_npc

            else: continue
            
            # --------------------------------------------- #
            # Get attack damage
            if 'null' in player['damage_targets']:
                attack_targets = player['damage_targets']['null']
                attack_damage = sum([attack_targets[target] for target in attack_targets if target in oppositing_npc])
            else:
                attack_damage = 0

            # --------------------------------------------- #
            # Get spell damage
            spell_damage = 0
            for spell in player['damage_targets']:
                if spell == 'null': continue
                for spell_target in player['damage_targets'][spell]:
                    if spell_target in oppositing_npc:
                        spell_damage += player['damage_targets'][spell][spell_target]

            # --------------------------------------------- #
            player_stats.gold = player['total_gold']
            player_stats.xp = player['total_xp']
            player_stats.kills = player['kills']
            player_stats.deaths = player['deaths']
            player_stats.assists = player['assists']
            player_stats.last_hits = player['lh_t'][-1]
            
            try:
                player_stats.last_hits_10 = player['lh_t'][10]
            except IndexError:
                player_stats.last_hits_10 = player['lh_t'][-1]

            try:
                player_stats.denies_5 = player['dn_t'][5]
            except IndexError:
                player_stats.denies_5 = player['dn_t'][-1]

            try:
                player_stats.denies_10 = player['dn_t'][10]
            except IndexError:
                player_stats.denies_10 = player['dn_t'][-1]

            player_stats.roshan_kills = player['roshan_kills']
            player_stats.tower_damage = player['tower_damage']
            player_stats.attack_damage = attack_damage
            player_stats.spell_damage = spell_damage
            player_stats.stuns = player['stuns']
            player_stats.heal = player['hero_healing']

            player_stats.obs_placed = player['obs_placed']
            player_stats.sen_placed = player['sen_placed']
            player_stats.camps_stacked = player['camps_stacked']
            player_stats.creeps_stacked = player['creeps_stacked']

            # --------------------------------------------- #
            obj.gold += player['total_gold']
            obj.xp += player['total_xp']
            obj.kills += player['kills']
            obj.deaths += player['deaths']
            obj.assists += player['assists']
            obj.last_hits += player['lh_t'][-1]

            try:
                obj.last_hits_10 += player['lh_t'][10]
            except IndexError:
                obj.last_hits_10 += player['lh_t'][-1]

            try:
                obj.denies_5 += player['dn_t'][5]
            except IndexError:
                obj.denies_5 += player['dn_t'][-1]

            try:
                obj.denies_10 += player['dn_t'][10]
            except IndexError:
                obj.denies_10 += player['dn_t'][-1]

            obj.roshan_kills += player['roshan_kills']
            obj.tower_damage += player['tower_damage']
            obj.attack_damage += attack_damage
            obj.spell_damage += spell_damage
            obj.stuns += player['stuns']
            obj.heal += player['hero_healing']

            obj.obs_placed += player['obs_placed']
            obj.sen_placed += player['sen_placed']
            obj.camps_stacked += player['camps_stacked']
            obj.creeps_stacked += player['creeps_stacked']

            # --------------------------------------------- #

            player_overview = _typing.property.PlayerOverview(
                slot=player['player_slot'],
                stats=player_stats,
                time_series=None)
            overview.players.append(player_overview)

        radiant_team_overview = _typing.property.TeamOverview(
            stats=radiant_team_stats,
            time_series=None)
        overview.teams.append(radiant_team_overview)

        dire_team_overview = _typing.property.TeamOverview(
            stats=dire_team_stats,
            time_series=None)
        overview.teams.append(dire_team_overview)

        return overview


class PropertyParser(DotaconstantsBase):
    """Property matches to `pd.DataFrame` parser"""
    # def __init__(self):
        # leagues = pd.read_csv('scarpe/output/leagues_liquid.csv', sep='\t').drop_duplicates('league_id').set_index('league_id')
        # matches = pd.read_csv('scarpe/output/matches_liquid.csv', sep='\t').drop_duplicates('match_id').set_index('match_id')

    def __call__(self, matches: list[_typing.property.Match | dict] | _typing.property.Match | dict) -> pd.DataFrame:
        """Parse property matches to `pd.DataFrame`"""
        if type(matches) != list:
            matches = [matches]

        if isinstance(matches[0], _typing.property.Match):
            matches = [m.dict() for m in matches]

        matches: list[dict]

        new_rows = []
        df = pd.DataFrame(matches)
        df['leavers'] = 0

        for idx in df.index:
            c = {}
            
            # ----------------------------------------------------------- #
            # league
            l = df.loc[idx, 'league']
            if l:
                l = {f"league_{k}":v for k,v in l.items()}
                c.update(l)
                        
            # ----------------------------------------------------------- #
            # players
            players = df.loc[idx, 'players']
            r_players = {}
            for p in players['radiant']:
                if p['leaver_status'] > 1:
                    df.loc[idx, 'leavers'] += 1
                
                r_players.update(
                    {f"{p['slot']}_{k}":v for k,v in p.items() if k != 'slot'}
                )

            d_players = {}
            for p in players['dire']:
                if p['leaver_status'] > 1:
                    df.loc[idx, 'leavers'] += 1

                d_players.update(
                    {f"{p['slot']}_{k}":v for k,v in p.items() if k != 'slot'}
                )
            c.update(r_players)
            c.update(d_players)

            # ----------------------------------------------------------- #
            # teams
            t = df.loc[idx, 'teams']
            if t:
                r = {f"r_team_{k}":v for k,v in t['radiant'].items()}
                d = {f"d_team_{k}":v for k,v in t['dire'].items()}
                c.update(r)
                c.update(d)
            
            # ----------------------------------------------------------- #
            # overview - teams 
            t = df.loc[idx, 'overview']['teams']
            r = {f"r_{k}":v for k,v in t[0]['stats'].items()}
            d = {f"d_{k}":v for k,v in t[1]['stats'].items()}
            c.update(r)
            c.update(d)

            # ----------------------------------------------------------- #
            # overview - players
            r_players = {}
            d_players = {}
            players_overview = df.loc[idx, 'overview']['players']
            for p in players_overview:
                if p['slot'] in self.RADIANT_SIDE:
                    r_players.update(
                        {f"{p['slot']}_{k}":v for k,v in p['stats'].items()}
                    )
                elif p['slot'] in self.DIRE_SIDE:
                    d_players.update(
                        {f"{p['slot']}_{k}":v for k,v in p['stats'].items()}
                    )  


            c.update(r_players)
            c.update(d_players)

            # ----------------------------------------------------------- #
            new_rows.append(c)

        df = pd.concat(
            objs=[df.drop(['league', 'players', 'teams'], axis=1),  pd.DataFrame(new_rows)], 
            axis=1
        ).sort_values(by='start_time').drop_duplicates('match_id')
        return df
