import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt; plt.style.use('seaborn')


def comparison_curve(df):
    """ df columns: y_true, y_pred, r_odd, d_odd, ID"""
    values = df.values

    b_curve = [0]
    m_curve = [0]
    for game in values:
        # Chance to win Radiant/Dire
        b_r, b_d = 1/game[2], 1/game[3]
        m_r, m_d = game[1], 1-game[1]

        if game[0] == 1:
            if b_r > b_d: b_curve.append(b_curve[-1] + b_r)
            else: b_curve.append(b_curve[-1] - b_r)

            if m_r > m_d: m_curve.append(m_curve[-1] + m_r)
            else: m_curve.append(m_curve[-1] - m_r)

        elif game[0] == 0:
            if b_r < b_d: b_curve.append(b_curve[-1] + b_d)
            else: b_curve.append(b_curve[-1] - b_d)

            if m_r < m_d: m_curve.append(m_curve[-1] + m_d)
            else: m_curve.append(m_curve[-1] - m_d)

    return b_curve, m_curve


def against_book_curve(df):
    """ df columns: y_true, y_pred, r_odd, d_odd, ID"""
    values = df.values

    bank = [100]
    for game in values:
        # Odds Radiant/Dire
        r_odd, d_odd = game[2], game[3]
        # Chance to win Radiant/Dire
        r_chance, d_chance = game[1], 1-game[1]

        # Place bet on raidnat
        if r_chance >= d_chance:
            # 10% of babk 
            bet = 10 * r_chance
            # Add a profit
            if game[0] == 1: bank.append(bank[-1] + bet*r_odd-bet)
            else: bank.append(bank[-1] - bet)

        elif r_chance < d_chance:
            bet = 10 * d_chance
            if game[0] == 0: bank.append(bank[-1] + bet*d_odd-bet)
            else: bank.append(bank[-1] - bet)
    return bank


def against_me_curve(df: pd.DataFrame):
    """ df columns: y_true, y_pred, r_odd, d_odd, ID"""
    values = df.values

    bank = [100]
    for game in values:
        # Odds Radiant/Dire
        r_odd, d_odd = 0.95/game[1], 0.95/(1-game[1])
        # Chance to win Radiant/Dire
        r_chance, d_chance = 0.95/game[2], 0.95/game[3]
        
        # Place bet on raidnat
        if r_chance >= d_chance:
            # 10% of babk 
            bet = 10 * r_chance
            # Add a profit
            if game[0] == 1: bank.append(bank[-1] + bet*r_odd-bet)
            else: bank.append(bank[-1] - bet)
                
        elif r_chance < d_chance:
            bet = 10 * d_chance
            if game[0] == 0: bank.append(bank[-1] + bet*d_odd-bet)
            else: bank.append(bank[-1] - bet)

    return bank


def smooth_values(x) -> pd.DataFrame:
    _range = np.arange(len(x))
    df1 = pd.DataFrame(
        data=zip(_range, x),
        columns=['timepoint', 'value'])
    df2 = pd.DataFrame(
        data=zip(_range, np.interp(_range, [0, len(x)-1], [x[0], x[-1]])),
        columns=['timepoint', 'value'])
    return pd.concat([df1, df2]).reset_index()


def profit_curves(df: pd.DataFrame, cf = 0.1):
    """ df columns: y_true, y_pred, r_odd, d_odd, ID"""
    values = df.values
    def profit_curve_1():
        bank = [100]
        B = bank[-1]
        for game in values:
            r_odd, d_odd = game[2], game[3]
            r_val, d_val = 1/game[1], 1/(1-game[1])
            if r_val < r_odd:
                bet = B * cf * (1/r_val)
                if game[0] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                else: bank.append(bank[-1] - bet)
            if d_val < d_odd:
                bet = B * cf * (1/d_val)
                if game[0] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                else: bank.append(bank[-1] - bet)
            B = bank[-1]
            # if B < bank[0]:
            #     B = bank[0]
        return bank
    
    def profit_curve_2():
        bank = [100]
        B = bank[-1]
        for game in values:
            r_odd, d_odd = game[2], game[3]
            r_val, d_val = 1/game[1], 1/(1-game[1])
            if r_val < r_odd:
                bet = B * cf * (1/r_val)
                if game[0] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                else: bank.append(bank[-1] - bet)
            if d_val < d_odd:
                bet = B * cf * (1/d_val)
                if game[0] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                else: bank.append(bank[-1] - bet)
            if bank[-1] > B:
                B = bank[-1]
        return bank
    
    return profit_curve_1(), profit_curve_2()


def metric_ts(df: pd.DataFrame, stride=8, window=None):
    curves = []
    if window is None:
        window = len(df)//2
        
    w = 0
    while w < len(df)-window+stride:
        inf = BinaryINF(
            df.iloc[w:w+window]
        )
        curves.append(
            [
                inf.accuracy,
                inf.auc,
                inf.precission,
                inf.recall,
            ]
        )
        w+=stride
    return np.array(curves)


def plot_charts(DF: pd.DataFrame):
    DF = DF[~(DF[['r_odd', 'd_odd']].values == False).sum(axis=1, dtype='bool')]
    DF.dropna(inplace=True)
    
    bk = '#4C72B0'
    me = '#4ba35b'
    linewidth = 2
    fontsize  = 15
    labelsize = int(fontsize*0.8)
    
    fig, axes = plt.subplots(3, 2, figsize=(24, 15))
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]
    ax4 = axes[2, 0]
    ax5 = axes[2, 1]
    
    bank_df = pd.DataFrame()
    for idx, cf in enumerate([0.1, 0.15]):
        bank1, bank2 = profit_curves(DF, cf)
        bank1_df = smooth_values(bank1)
        bank2_df = smooth_values(bank2)
        bank1_df['bank'] = 'current'
        bank2_df['bank'] = 'max'
        bank1_df['cf'] = cf
        bank2_df['cf'] = cf
        bank_df = pd.concat([bank_df, bank1_df, bank2_df])
    bank_df['value'] = bank_df['value'] - 100
    bank_df.reset_index(inplace=True)
    
    my_play = against_book_curve(DF)
    book_play = against_me_curve(DF)
    b_curve, m_curve = comparison_curve(DF)
    bank1, bank2 = profit_curves(DF)
    
    palette = sns.color_palette("flare", as_cmap=True)
    sns.lineplot(
        data=bank_df, x="timepoint", y="value", hue="cf", 
        style="bank", palette=palette, ax=ax1,
    )
    [t.set_fontsize(fontsize) for t in ax1.legend_.texts]
    ax1.axhline(0, color='black', alpha=0.5)
    
    ax0.plot(
        np.array(m_curve)-np.array(b_curve),
        c=me, 
        linewidth=linewidth
    )
    ax0.tick_params(labelsize=labelsize)
    ax0.set_title('Comparison of forecasts', fontsize=fontsize)
    ax0.set_ylabel('Score', fontsize=fontsize)
    ax0.set_xlabel('Games', fontsize=fontsize)

    ax2.plot(m_curve, c=me, linewidth=linewidth)
    ax2.plot(b_curve, c=bk, linewidth=linewidth)
    ax2.tick_params(labelsize=labelsize)
    ax2.legend(['me', 'bookmaker (include margin)'], fontsize=fontsize)
    ax2.set_title('Comparison of forecasts', fontsize=fontsize)
    ax2.set_ylabel('Score', fontsize=fontsize)
    ax2.set_xlabel('Games', fontsize=fontsize)


    book_df = smooth_values(book_play)
    my_df = smooth_values(my_play)
    sns.lineplot(
        data=my_df, 
        x="timepoint", 
        y="value", 
        linewidth=linewidth, 
        color=me,
        alpha=0.8,
        ax=ax5)
    ax5.axhline(np.mean(my_play), alpha=0.9, c=me, linestyle='--')
    ax5.axhline(100, alpha=0.25, c='black')
    ax5.tick_params(labelsize=labelsize)
    ax5.set_title("My play against book odds", fontsize=fontsize)
    ax5.set_xlabel('Games', fontsize=fontsize)
    ax5.set_ylabel('Bank profit (in %)', fontsize=fontsize)

    sns.lineplot(
        data=book_df, 
        x="timepoint", 
        y="value", 
        linewidth=linewidth, 
        color=bk,
        alpha=0.8,
        ax=ax4)
    ax4.axhline(np.mean(book_play), alpha=0.9, c=bk, linestyle='--')
    ax4.axhline(100, alpha=0.25, c='black')
    ax4.tick_params(labelsize=labelsize)
    ax4.set_title("Book play against my odds (include margin):", fontsize=fontsize)
    ax4.set_xlabel('Games', fontsize=fontsize)
    ax4.set_ylabel('Bank profit (in %)', fontsize=fontsize)

    bank1_df = smooth_values(bank1)
    bank2_df = smooth_values(bank2)
    bank1_df['bank'] = 1
    bank2_df['bank'] = 2
    bank_df = pd.concat([bank1_df, bank2_df]).reset_index()

    palette = sns.color_palette("flare", as_cmap=True)
    sns.lineplot(
        data=bank_df, 
        x="timepoint", 
        y="value", 
        hue="bank",
        linewidth=linewidth, 
        alpha=0.8,
        palette=palette,
        ax=ax3)
    ax3.tick_params(labelsize=labelsize)
    ax3.set_title("Bets simulation", fontsize=fontsize)
    ax3.set_xlabel('Games', fontsize=fontsize)
    ax3.set_ylabel('Bank profit (in %)', fontsize=fontsize)
    ax3.legend(['Bets by current bank', 'Bets by max bank'], fontsize=fontsize)
    plt.show()
    
    df = pd.read_csv('../DataParse/out/LeagueDataFrame.csv')
    first_day = df[df['match_id'] == DF.iloc[0]['ID']]['start_time'].values[0]
    last_day = df[df['match_id'] == DF.iloc[-1]['ID']]['start_time'].values[0]
    days_gone = (last_day - first_day)/60/60/24
    p = bank1[-1] - 100
    print('Range in days :', days_gone)
    print()
    print('First match :', int(DF.iloc[0]['ID']))
    print('Last match :', int(DF.iloc[-1]['ID']))
    print()
    print('X to bank for month :')
    print((30*p/days_gone + 100)/100)