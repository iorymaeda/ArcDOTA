"""Prepare data to train: convert matches to dataframe, split to train/val/test"""

import re
import sys
import yaml
import json
import time
import logging
import pathlib
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo 
import pandas as pd

import utils


RADIANT_SIDE = [0, 1, 2, 3, 4]
DIRE_SIDE = [128, 129, 130, 131, 132]

def reopen():
    global client
    
    try: client.close()
    except: pass
    client = pymongo.MongoClient(
        "mongodb://localhost:27017/", 
        maxPoolSize=50
    )

def load_table(table):
    reopen()
    _matches = client['ParsedMatches'][table].find()
    return list(_matches)

def log(t):
    if __name__ == "__main__":
        if __verbose: logging.info(t)
    else:
        if __verbose: print(t)

def convert(d: str):
    d = json.loads(d)
    
    if isinstance(d, list):
        return None
    
    new_d = {}
    for k, v in d.items():
        try:
            v = int(v.replace(',', '').replace('$', ''))
            _v = True
        except Exception: continue

        try:
            k = re.sub(r'[a-zA-Z]+', '', k)
            if '-' in k:
                a1, a2 = map(int, k.split('-'))
                for place in range(a1, a2+1):
                    new_d[place] = v
            else:
                k = int(k)
                new_d[k] = v
        except Exception as e: pass
            
    return new_d


def run(verbose=True, months=24):
    def fill_league_features(df: pd.DataFrame):
        for row in df.iterrows():
            idx, row = row
            match_id = row['match_id']
            league_id = row['league_id']

            if (match_id in matches_liquid.index and 
                league_id in leagues_liquid.index):

                features = dict(matches_liquid.loc[match_id])
                features['is_dpc'] = dict(leagues_liquid.loc[league_id])['is_dpc']

                for f in features:
                    df.loc[idx, f] = features[f]
        return df

    global __verbose
    __verbose = verbose

    with open('../configs/train.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    with open('../configs/features.yaml', 'r') as stream:
        f_config = yaml.safe_load(stream)

    parser = utils.parsers.PropertyParser()
    evaluator = utils.Evaluator()

    matches_liquid = pd.read_csv('../scarpe/output/matches_liquid.csv', sep='\t').drop_duplicates('match_id').set_index('match_id')
    leagues_liquid = pd.read_csv('../scarpe/output/leagues_liquid.csv', sep='\t').drop_duplicates('league_id').set_index('league_id')
    leagues_liquid['prize_pool'] = leagues_liquid['prize_pool'].map(lambda x: x.replace('\xa0', ''))
    leagues_liquid['dpc_points'] = leagues_liquid['dpc_points'].map(lambda x: x.replace('\xa0', ''))
    leagues_liquid['prize_pool'] = leagues_liquid['prize_pool'].map(lambda x: convert(x))
    leagues_liquid['dpc_points'] = leagues_liquid['dpc_points'].map(lambda x: convert(x))

    train_last_match = None
    for mode in ['league', 'public']:
        tokenizer = utils.Tokenizer(mode=mode)
        scaler = utils.scalers.DotaScaler(utils._typing.property.FEATURES)

        log(f'# ----------------------------------------------------------- #')
        log(f'start parse {mode}')

        # ----------------------------------------------------------- #
        # Load and parse
        matches = load_table(f'{mode}Matches')
        log(f'matches has been loaded from db')
        log(f'number of matches: {len(matches)}')
        
        df = parser(matches).sort_values(by='start_time').drop_duplicates('match_id')
        log(f'number of matches after parser: {len(df)}')
        with utils.development.suppress(KeyError):
            df.drop(['_id'], inplace=True, axis=1)

        # ----------------------------------------------------------- #
        # Purge old games //TODO: FIX THIS, MAKE THIS FLEXIBLE
        if mode == 'league':
            df = df[df['start_time'] > int(time.time()) - 3600*24*30*months]
            log(f'number of matches after purge old games: {len(df)}')

        # ----------------------------------------------------------- #
        # Evaluate and drop
        df = evaluator(df)
        log(f'number of matches after evaluator: {len(df)}')

        if mode == 'league':
            df = df[df['isleague']]
        else:
            df = df[~df['isleague']]
        log(f'number of matches after drop league/nonleague: {len(df)}')
        
        df = df[~(df['leavers'] > 0)]
        log(f'number of matches after drop leavers: {len(df)}')

        [ df[f'{s}_rank_tier'].fillna(0, inplace=True) for s in RADIANT_SIDE+DIRE_SIDE ]
        [ df[f'{s}_account_id'].fillna(0, inplace=True) for s in RADIANT_SIDE+DIRE_SIDE ]
        df['region'].fillna(0, inplace=True)
        if mode == 'public':
            df['r_team_id'].fillna(0, inplace=True)
            df['d_team_id'].fillna(0, inplace=True)

            df['series_type'].fillna(0, inplace=True)
            df['series_id'].fillna(0, inplace=True)

            df['league_prize_pool'].fillna(0, inplace=True)
            df['league_tier'].fillna(0, inplace=True)
            df['league_name'].fillna(0, inplace=True)
            df['league_id'].fillna(0, inplace=True)
            
        elif mode == 'league' and f_config['league']['features']['tabular']['grid']:
            df = fill_league_features(df)
            
        df['league_prize_pool'] = df['league_prize_pool'].map(utils.scalers.vectorize_prize_pool)
        df.dropna(inplace=True)
        log(f'number of matches after drop na: {len(df)}')

        # ----------------------------------------------------------- #
        # Split train/val/test
        log(f'split matches...')
        l = len(df)
        t_size = config[mode]['split']['test']
        if t_size > 1:
            test_df = df[-t_size:]
            df = df[:-t_size]
        elif 0 < t_size < 1:
            test_df = df[-int(l*t_size):]
            df = df[:-int(l*t_size)]
            
        l = len(df)
        v_size = config[mode]['split']['val']
        if v_size > 1:
            val_df = df[-v_size:]
            df = df[:-v_size]
        elif 0 < v_size < 1:
            val_df = df[-int(l*v_size):]
            df = df[:-int(l*v_size)]
            
        train_df = df
        log(f'train size: {len(train_df)}')
        if v_size > 0:
            log(f'val size: {len(val_df)}')
        if t_size > 0:
            log(f'test size: {len(test_df)}')

        # Games that played after last `league` game, we should remove from train data
        if mode == 'public':
            games_to_remove = train_df['start_time'] >= train_last_match
            if games_to_remove.sum() > 0:
                if v_size > 0:
                    val_df = pd.concat([train_df[games_to_remove], val_df]).sort_values(by='start_time')
                else:
                    val_df = train_df[games_to_remove]

                # increameant v_size, to proccesing it and save
                v_size += 1
                train_df = train_df[~games_to_remove]
                log(f'train size after remove leaked games: {len(train_df)}')
                log(f'val size after remove leaked games: {len(val_df)}')

        # ----------------------------------------------------------- #
        # Save raw data
        log(f'save...')
        train_df.reset_index(drop=True).to_csv(f'output/{mode}/raw_train_df.csv')
        if v_size > 0:
            val_df.reset_index(drop=True).to_csv(f'output/{mode}/raw_val_df.csv')
        if t_size > 0:
            test_df.reset_index(drop=True).to_csv(f'output/{mode}/raw_test_df.csv')

        # ----------------------------------------------------------- #
        # Tokenize
        log(f'tokenize...')
        tokenizer.fit(train_df)

        if mode == 'public':
            train_df = tokenizer.tokenize(train_df, players=True, teams=True)
            if v_size > 0:
                val_df = tokenizer.tokenize(val_df, players=True, teams=True)
            if t_size > 0:
                test_df = tokenizer.tokenize(test_df, players=True, teams=True )

        # This is comented cause, we drop this unevaluated matches in train pipelines
        # if mode == 'league':
        #     train_df = tokenizer.evaluate(train_df)
        #     val_df = tokenizer.evaluate(val_df)
        #     test_df = tokenizer.evaluate(test_df)

        #     log(f'number of matches after tokinize train_df: {len(train_df)}')
        #     log(f'number of matches after tokinize   val_df: {len(val_df)}')
        #     log(f'number of matches after tokinize  test_df: {len(test_df)}')
        tokenizer.save(f'output/tokenizer_{mode}.pkl')

        # ----------------------------------------------------------- #
        log(f'scaling...')
        scaler.fit(train_df, 'teams')
        scaler.fit(train_df, 'players')

        # //TODO: put it to configs
        train_df = scaler.transform(train_df, 'yeo-johnson', mode='both')
        if v_size > 0:
            val_df = scaler.transform(val_df, 'yeo-johnson', mode='both')   
        if t_size > 0:
            test_df = scaler.transform(test_df, 'yeo-johnson', mode='both')

        scaler.save(f'output/scaler_{mode}.pkl')
        # ----------------------------------------------------------- #
        log(f'save...')
        train_df.reset_index(drop=True).to_csv(f'output/{mode}/train_df.csv', index=False)
        if v_size > 0:
            val_df.reset_index(drop=True).to_csv(f'output/{mode}/val_df.csv', index=False)
        if t_size > 0:
            test_df.reset_index(drop=True).to_csv(f'output/{mode}/test_df.csv', index=False)

        if mode == 'league':
            train_last_match = train_df['start_time'].max()

if __name__ == "__main__":
    logging.basicConfig(filename="logs/prepare.log", level=logging.INFO)
    run()