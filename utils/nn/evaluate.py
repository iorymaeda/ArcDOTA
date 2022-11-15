import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class BinaryMetrics():
    def __init__(self, df: pd.DataFrame = None, y_true=None, y_pred=None):
        assert df is not None or (y_true is not None and y_pred is not None), 'Provide at least one argument'

        if df is not None:
            y_true, y_pred = df['y_true'].values, df['y_pred'].values

        self.y_true, self.y_pred = y_true, y_pred
        self.df = df
        
        self.balance = y_true.mean()
        self.accuracy = self._accuracy(y_true, y_pred)
        self.balanced_accuracy = self._balanced_accuracy(y_true, y_pred)
        self.precission = self._precision(y_true, y_pred)
        self.recall = self._recall(y_true, y_pred)
        self.confusion_matrix = self._confusion_matrix(y_true, y_pred)
        self.fpr, self.tpr, self.threshold, self.auc, self.eer = self._roc_auc(y_true, y_pred)
        self.log_loss = self._log_loss(y_true, y_pred)
        self.mae = self._mae(y_true, y_pred)
        self.mse = self._mse(y_true, y_pred)

        self.summary = {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy, 
            'precission': self.precission, 
            'recall': self.recall, 
            'auc': self.auc, 
            'eer': self.eer, 
            'log_loss': self.log_loss, 
            'mae': self.mae, 
            'mse': self.mse, 
        }

    def _roc_auc(self, y_true, y_pred):
        fpr, tpr, threshold = metrics.roc_curve(y_true,  y_pred)

        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer1, eer2 = fpr[np.nanargmin(np.absolute((fnr - fpr)))], fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = (eer1+eer2)/2
        
        auc = metrics.roc_auc_score(y_true, y_pred)
        return fpr, tpr, threshold, auc, eer
        
    def _accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred.round())
    
    def _balanced_accuracy(self, y_true, y_pred):
        return metrics.balanced_accuracy_score(y_true, y_pred.round())
    
    def _precision(self, y_true, y_pred):
        """Precision   можно   интерпретировать  как   долю 
        объектов,  названных классификатором положительными 
        и при этом действительно являющимися положительными
        """
        return metrics.precision_score(y_true, np.around(y_pred))
    
    def _recall(self, y_true, y_pred):
        """recall показывает, какую долю объектов 
        положительного  класса  из  всех объектов 
        положительного  класса   нашел   алгоритм
        """
        return metrics.recall_score(y_true, np.around(y_pred))
    
    def _confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, np.around(y_pred))
    
    def _log_loss(self, y_true,  y_pred):
        return metrics.log_loss(y_true, y_pred)
    
    def _mae(self, y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)
    
    def _mse(self, y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        s = f"--------------------------------------\n"
        s+= f"accuracy   = {self.accuracy}\n"
        s+= f"b_accuracy = {self.balanced_accuracy}\n"
        s+= f"AUC        = {self.auc}\n\n"
        
        s+= f"log_loss   = {self.log_loss}\n"
        s+= f"mae        = {self.mae}\n"
        s+= f"mse        = {self.mse}\n"
        s+= f"--------------------------------------"
        return s

class BookEvaluator:
    def __init__(self, df: pd.DataFrame):
        """ df columns: match_id, y_true, book_name, r_odd, d_odd, r_pred, d_pred, y_pred"""
        self.df = df
        self.metrics_m = BinaryMetrics(df)
        
        df = df.copy()
        df['y_pred'] = df['r_pred']
        self.metrics_b = BinaryMetrics(df)
        
    @staticmethod
    def comparison_curve(df):
        """ df columns: y_true, y_pred, r_odd, d_odd, ID"""
        values = df.values

        b_curve = [0]
        m_curve = [0]
        for game in df.iterrows():
            game = game[1]
            # Chance to win Radiant/Dire
            b_r, b_d = 1/game['r_odd'], 1/game['d_odd']
            m_r, m_d = game['y_pred'], 1-game['y_pred']

            if game['y_true'] == 1:
                if b_r > b_d: b_curve.append(b_curve[-1] + b_r)
                else: b_curve.append(b_curve[-1] - b_r)

                if m_r > m_d: m_curve.append(m_curve[-1] + m_r)
                else: m_curve.append(m_curve[-1] - m_r)

            elif game['y_true'] == 0:
                if b_r < b_d: b_curve.append(b_curve[-1] + b_d)
                else: b_curve.append(b_curve[-1] - b_d)

                if m_r < m_d: m_curve.append(m_curve[-1] + m_d)
                else: m_curve.append(m_curve[-1] - m_d)

        return b_curve, m_curve
    
    @staticmethod
    def against_book_curve(df):
        """Book play against my raw odds, with simple strategy"""
        bank = [100]
        for game in df.iterrows():
            B = bank[-1]
            game = game[1]
            # Odds Radiant/Dire
            r_odd, d_odd = game['r_odd'], game['d_odd']
            # Chance to win Radiant/Dire
            r_chance, d_chance = game['y_pred'], 1-game['y_pred']
            
            # Place bet on raidnat
            if r_chance >= d_chance:
                # 10% of babk 
                bet = 0.1 * B * r_chance
                # Add a profit
                if game['y_true'] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                else: bank.append(bank[-1] - bet)

            elif r_chance < d_chance:
                bet = 0.1 * B * d_chance
                if game['y_true'] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                else: bank.append(bank[-1] - bet)
                
        return bank
    
    @staticmethod
    def against_me_curve(df):
        """Me play against book raw odds, with simple strategy"""
        bank = [100]
        for game in df.iterrows():
            B = bank[-1]
            game = game[1]
            # Odds Radiant/Dire
            r_odd, d_odd = 0.95/game['y_pred'], 0.95/(1-game['y_pred'])
            # Chance to win Radiant/Dire
            r_chance, d_chance = game['r_pred'], game['d_pred']

            # Place bet on raidnat
            if r_chance >= d_chance:
                # 10% of babk 
                bet = 0.1 * B * r_chance
                # Add a profit
                if game['y_true'] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                else: bank.append(bank[-1] - bet)

            elif r_chance < d_chance:
                bet = 0.1 * B * d_chance
                if game['y_true'] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                else: bank.append(bank[-1] - bet)

        return bank
    
    @staticmethod
    def smooth_values(x) -> pd.DataFrame:
        _range = np.arange(len(x))
        df1 = pd.DataFrame(
            data=zip(_range, x),
            columns=['timepoint', 'value'])
        df2 = pd.DataFrame(
            data=zip(_range, np.interp(_range, [0, len(x)-1], [x[0], x[-1]])),
            columns=['timepoint', 'value'])
        return pd.concat([df1, df2]).reset_index()
    
    @staticmethod
    def profit_curves(df, cf = 0.1):
        def profit_curve_1():
            bank = [100]
            for game in df.iterrows():
                B = bank[-1]
                game = game[1]
                r_odd, d_odd = game['r_odd'], game['d_odd']
                r_val, d_val = 1/game['y_pred'], 1/(1-game['y_pred'])
                
                # if abs(0.5-game['y_pred']) < 0.05: continue
                
                if r_val < r_odd:
                    bet = B * cf * (1/r_val)
                    if game['y_true'] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                    else: bank.append(bank[-1] - bet)
                if d_val < d_odd:
                    bet = B * cf * (1/d_val)
                    if game['y_true'] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                    else: bank.append(bank[-1] - bet)
            return bank

        def profit_curve_2():
            bank = [100]
            B = bank[-1]
            for game in df.iterrows():
                game = game[1]
                r_odd, d_odd = game['r_odd'], game['d_odd']
                r_val, d_val = 1/game['y_pred'], 1/(1-game['y_pred'])
                
                # if abs(0.5-game['y_pred']) < 0.05: continue
                
                if r_val < r_odd:
                    bet = B * cf * (1/r_val)
                    if game['y_true'] == 1: bank.append(bank[-1] + bet*r_odd-bet)
                    else: bank.append(bank[-1] - bet)
                if d_val < d_odd:
                    bet = B * cf * (1/d_val)
                    if game['y_true'] == 0: bank.append(bank[-1] + bet*d_odd-bet)
                    else: bank.append(bank[-1] - bet)
                if bank[-1] > B:
                    B = bank[-1]
            return bank

        return profit_curve_1(), profit_curve_2()
    
    def metric_ts(self, df=None, stride=4, window=None, book=False):
        if df is None: 
            df = self.df
        if window is None:
            window = len(self.df)//2
        
        df = self.df.copy()
        if book:
            df['y_pred'] = df['r_pred']

        w = 0
        curves = []
        legend = ['accuracy', 'auc', 'log_loss']
        while w < len(df)-window+stride:
            inf = BinaryMetrics(df.iloc[w:w+window])
            curves.append([inf.accuracy, inf.auc, inf.log_loss])
            w+=stride
            
        return np.array(curves)*100, legend
    
    def calibration_curve(self, df=None, ax=None, book=False, figsize=(16, 8)):
        if df is None: 
            df = self.df
        if ax is None: 
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        if book:
            df = df.copy()
            df['y_pred'] = df['r_pred']
            
        y_pred, y_true = df['y_pred'], df['y_true']
        
        ax.set_title("Calibration curve", fontsize=14)
        display = CalibrationDisplay.from_predictions(
            y_true,
            y_pred, 
            n_bins=10,
            name="Ensembling models",
            ax=ax,
        )

        major_ticks = np.arange(0, 1+1e-6, 0.1)
        minor_ticks = np.arange(0, 1+1e-6, 0.025)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        _ax = ax.twinx()
        sns.distplot(y_pred, ax=_ax, color='orange', hist=True, kde=False, rug=True)
        
    def probplot(self, ax=None, y_true=None, y_pred=None, fontsize=14):
        def prob_mean_loss(y_true, y_pred):
            arr = y_true - y_pred
            return (arr.sum()/arr.shape[0])*100
        
        if y_true is None:
            y_true = self.df['y_true']
        if y_pred is None:
            y_pred = self.df['y_pred']
            
        df = pd.DataFrame()
        window = np.arange(0., 1.1, 0.1)
        for w1,w2 in zip(window[:-1], window[1:]):
            bool_ = (y_pred > w1) & (y_pred < w2)
            
            p = y_pred[bool_]
            acc = metrics.accuracy_score(y_true[bool_], p.round())
            loss = prob_mean_loss(y_true[bool_], y_pred[bool_])
            
            p = p.mean()
            if (w2+w1)/2 < 0.5:
                column = f"{int((1-w1)*100)}-{int((1-w2)*100)}"
                df.loc['Accuracy', column] = acc * 100
                df.loc['Mean pred', column] = (1-p) * 100
                df.loc['Loss %', column]   = -loss
                df.loc['Samples', column]  = bool_.sum()
            else:
                column = f"{int(w1*100)}-{int(w2*100)}"
                df.loc['Accuracy', column] = acc * 100
                df.loc['Mean pred', column] = p * 100
                df.loc['Loss %', column]   = loss
                df.loc['Samples', column]  = bool_.sum()
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
        ax.bar(df.columns, df.loc['Samples'], color='g', alpha=0.25)
        #ax.bar(df.columns, df.loc['Samples'].max() - (df.loc['Accuracy']/100)*df.loc['Samples'].max(), color='g', alpha=0.25)

        loss = df.loc['Loss %']#*df.loc['Samples'].max()/100
        ax.bar(df.columns[loss <= 0], loss[loss <= 0], color='b', alpha=0.2)
        ax.bar(df.columns[loss >= 0], -loss[loss >= 0], color='r', alpha=0.2)
        
        ax.legend(['num of predictions', 'overconfidence', 'underconfidence'], fontsize=fontsize)
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        return df

def gambling_charts(DF: pd.DataFrame):
    """DF columns: r_odd, d_odd..."""
    DF = DF[~(DF[['r_odd', 'd_odd']].values == False).sum(axis=1, dtype='bool')]
    DF.dropna(inplace=True)
    
    bk = '#4C72B0'
    me = '#4ba35b'
    linewidth = 2
    fontsize  = 15
    labelsize = int(fontsize*0.8)
    palette   = sns.color_palette("flare", as_cmap=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(32, 16))
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]
    ax4 = axes[2, 0]
    ax5 = axes[2, 1]
    
    my_play = BookEvaluator.against_book_curve(DF)
    book_play = BookEvaluator.against_me_curve(DF)
        
    bank_df = pd.DataFrame()
    for idx, cf in enumerate([0.05, 0.1, 0.15]):
        _bank1, _bank2 = BookEvaluator.profit_curves(DF, cf)
        bank1_df = BookEvaluator.smooth_values(_bank1)
        bank2_df = BookEvaluator.smooth_values(_bank2)
        bank1_df['bank'] = 'current'
        bank2_df['bank'] = 'max'
        bank1_df['cf'] = cf
        bank2_df['cf'] = cf
        bank_df = pd.concat([bank_df, bank1_df, bank2_df])
    bank_df['value'] = bank_df['value'] - 100
    bank_df.reset_index(inplace=True)
    
    # ---------------------------------------------------------------------- #
    sns.lineplot(
        data=bank_df, x="timepoint", y="value", hue="cf", 
        style="bank", palette=palette, ax=ax1)
    ax1.set_title("Bets simulation with different params", fontsize=fontsize)
    [t.set_fontsize(fontsize) for t in ax1.legend_.texts]
    ax1.axhline(0, color='black', alpha=0.5)
    
    # ---------------------------------------------------------------------- #
    b_curve, m_curve = BookEvaluator.comparison_curve(DF)
    ax0.plot(np.array(m_curve)-np.array(b_curve), c=me, linewidth=linewidth)
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
    
    # ---------------------------------------------------------------------- #
    book_df = BookEvaluator.smooth_values(book_play)
    my_df = BookEvaluator.smooth_values(my_play)
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
    
    # ---------------------------------------------------------------------- #
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
    
    # ---------------------------------------------------------------------- #
    bank1, bank2 = BookEvaluator.profit_curves(DF)
    bank1_df = BookEvaluator.smooth_values(bank1)
    bank2_df = BookEvaluator.smooth_values(bank2)
    bank1_df['bank'], bank2_df['bank'] = 1, 2
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
    return fig

def compression_charts(dfM, dfB, evaluator, fontsize=14):
    """Comprassion against bookmaker prediction
    dfM: pd.DataFrame with raw prediction
    dfB: pd.DataFrame with prediction on odds corpus
    """
    n0, legend = evaluator.metric_ts(df=dfM, stride=8)
    n1, legend = evaluator.metric_ts(stride=8)
    n2, legend = evaluator.metric_ts(stride=8, book=True)

    fig, axes = plt.subplot_mosaic([['A', 'E'], 
                                    ['B', 'E'], 
                                    ['C', 'F'], 
                                    ['C', 'F']], figsize=(32, 16)) 
    #plt.subplot_mosaic("AABB;.DD.", figsize=(24, 8))

    # ---------------------------------------------- #
    # Metrics over time
    axes['A'].plot(n1)
    axes['A'].set_title('My metrics over time', fontsize=fontsize)

    axes['B'].plot(n2)
    axes['B'].set_title('Bookmaker metrics over time', fontsize=fontsize)
    fig.legend(legend, fontsize=int(fontsize*1.1), loc='center')

    # ---------------------------------------------- #
    # Bet simulation

    # Try different params
    bank_df = pd.DataFrame()
    for idx, cf in enumerate([0.05, 0.1, 0.15]):
        _bank1, _bank2 = BookEvaluator.profit_curves(dfB, cf)
        bank1_df = BookEvaluator.smooth_values(_bank1)
        bank2_df = BookEvaluator.smooth_values(_bank2)
        bank1_df['bank'] = 'current'
        bank2_df['bank'] = 'max'
        bank1_df['cf'] = cf
        bank2_df['cf'] = cf
        bank_df = pd.concat([bank_df, bank1_df, bank2_df])
    bank_df['value'] = bank_df['value']
    bank_df.reset_index(inplace=True)
    
    palette = sns.color_palette("flare", as_cmap=True)
    sns.lineplot(
        data=bank_df, x="timepoint", y="value", hue="cf", 
        style="bank", palette=palette, ax=axes['C'])
    [t.set_fontsize(fontsize) for t in axes['C'].legend_.texts]
    axes['C'].set_title("Bets simulation with different cf", fontsize=fontsize)
    axes['C'].axhline(100, color='black', alpha=0.5)
    axes['C'].set_xlabel('Games', fontsize=fontsize)
    axes['C'].set_ylabel('Bank profit (in %)', fontsize=fontsize)

    max_ = bank_df['value'].max()
    min_ = bank_df['value'].min()
    range_ = max_ - min_
    major_ticks = np.arange(min_, max_+1e-6, range_/16)
    minor_ticks = np.arange(min_, max_+1e-6, range_/8)
    axes['C'].set_yticks(major_ticks)
    axes['C'].set_yticks(minor_ticks, minor=True)

    # ---------------------------------------------------------------------- #
    # bank1, bank2 = BookEvaluator.profit_curves(dfM)
    # bank1_df = BookEvaluator.smooth_values(bank1)
    # bank2_df = BookEvaluator.smooth_values(bank2)
    # bank1_df['bank'] = 'Bets by current bank'
    # bank2_df['bank'] = 'Bets by max bank'
    # bank_df = pd.concat([bank1_df, bank2_df]).reset_index()
    
    # palette = sns.color_palette("flare", as_cmap=True)
    # sns.lineplot(
    #     data=bank_df, x="timepoint", y="value", hue="bank",
    #     alpha=0.8, palette=palette, ax=axes['D'])
    # [t.set_fontsize(fontsize) for t in axes['D'].legend_.texts]
    # axes['D'].tick_params(labelsize=int(fontsize*0.8))
    # axes['D'].set_title("Bets simulation", fontsize=fontsize)
    # axes['D'].set_xlabel('Games', fontsize=fontsize)
    # axes['D'].set_ylabel('Bank profit (in %)', fontsize=fontsize)
    # plt.show()

    # ---------------------------------------------- #
    # Calibration curves
    evaluator.calibration_curve(df=dfB, ax=axes['E'])
    axes['E'].set_title('Calibration curve My', fontsize=fontsize)
    axes['E'].set_xlabel('')

    evaluator.calibration_curve(df=dfB, ax=axes['F'], book=True)
    axes['F'].set_title('Calibration curve Book', fontsize=fontsize)
    return fig

