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
        log(f'# ----------------------------------------------------------- #')
        log(f'start parse {mode}')

        # ----------------------------------------------------------- #
        matches = load_table(f'{mode}Matches')
        log(f'matches has been loaded from db')
        log(f'number of matches: {len(matches)}')
        
        df = parser(matches)
        log(f'number of matches after parser: {len(df)}')

        # ----------------------------------------------------------- #
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
        df.sort_values(
            by='start_time', 
            inplace=True,
        )
        log(f'number of matches after drop na: {len(df)}')

        # ----------------------------------------------------------- #
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
        log(f'val size: {len(val_df)}')
        log(f'test size: {len(test_df)}')

        # ----------------------------------------------------------- #
        log(f'save...')
        train_df.reset_index(drop=True).to_json(f'output/{mode}/raw_train_df.json')
        val_df.reset_index(drop=True).to_json(f'output/{mode}/raw_val_df.json')
        test_df.reset_index(drop=True).to_json(f'output/{mode}/raw_test_df.json')

        # ----------------------------------------------------------- #
        log(f'tokinize...')
        tokenizer = utils.Tokenizer()
        tokenizer.fit(train_df)

        train_df = tokenizer.tokenize(train_df)
        val_df = tokenizer.tokenize(val_df)
        test_df = tokenizer.tokenize(test_df)

        if mode == 'league':
            train_df = tokenizer.evaluate(train_df)
            val_df = tokenizer.evaluate(val_df)
            test_df = tokenizer.evaluate(test_df)

            log(f'number of matches after tokinize train_df: {len(train_df)}')
            log(f'number of matches after tokinize   val_df: {len(val_df)}')
            log(f'number of matches after tokinize  test_df: {len(test_df)}')

        # ----------------------------------------------------------- #
        log(f'scaling...')
        scaler = utils.scalers.BaseScaler(utils._typing.property.FEATURES)
        scaler.fit(train_df)

        train_df = scaler.transform(train_df, 'minmax2')
        val_df = scaler.transform(val_df, 'minmax2')
        test_df = scaler.transform(test_df, 'minmax2')

        # ----------------------------------------------------------- #
        log(f'save...')
        train_df.reset_index(drop=True).to_json(f'output/{mode}/train_df.json')
        val_df.reset_index(drop=True).to_json(f'output/{mode}/val_df.json')
        test_df.reset_index(drop=True).to_json(f'output/{mode}/test_df.json')


if __name__ == "__main__":
    logging.basicConfig(filename="logs/prepare.log", level=logging.INFO)
    run()