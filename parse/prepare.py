# importing modules from parent folder
import sys; sys.path.append('../')
import yaml
import logging
 
import pymongo 

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
        
def run(verbose=True):
    global __verbose
    __verbose = verbose

    with open('../configs/train.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    
    parser = utils.parsers.PropertyParser()
    evaluator = utils.Evaluator()

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
        
        df = parser(matches)
        log(f'number of matches after parser: {len(df)}')
        with utils.development.suppress(KeyError):
            df.drop(['_id'], inplace=True, axis=1)

        # ----------------------------------------------------------- #
        # Purge old games //TODO: FIX THIS, MAKE THIS FLEXIBLE
        # df = df[df['start_time'] > 1577826000]
        # log(f'number of matches after purge old games: {len(df)}')

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

            df['league_prize_pool'].fillna(0, inplace=True)
            df['league_tier'].fillna(0, inplace=True)
            df['league_name'].fillna(0, inplace=True)
            df['league_id'].fillna(0, inplace=True)
        
        df.dropna(inplace=True)
        log(f'number of matches after drop na: {len(df)}')

        # ----------------------------------------------------------- #
        # Split train/val/test
        log(f'split matches...')
        l = len(df)
        t_size = config['split'][mode]['test']
        if t_size > 1:
            test_df = df[-t_size:]
            df = df[:-t_size]
        elif 0 < t_size < 1:
            test_df = df[-int(l*t_size):]
            df = df[:-int(l*t_size)]
            
        l = len(df)
        v_size = config['split'][mode]['val']
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

        # ----------------------------------------------------------- #
        # Save raw data
        log(f'save...')
        train_df.reset_index(drop=True).to_json(f'output/{mode}/raw_train_df.json')
        if v_size > 0:
            val_df.reset_index(drop=True).to_json(f'output/{mode}/raw_val_df.json')
        if t_size > 0:
            test_df.reset_index(drop=True).to_json(f'output/{mode}/raw_test_df.json')

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
        train_df.reset_index(drop=True).to_json(f'output/{mode}/train_df.json')
        if v_size > 0:
            val_df.reset_index(drop=True).to_json(f'output/{mode}/val_df.json')
        if t_size > 0:
            test_df.reset_index(drop=True).to_json(f'output/{mode}/test_df.json')


if __name__ == "__main__":
    logging.basicConfig(filename="logs/prepare.log", level=logging.INFO)
    run()