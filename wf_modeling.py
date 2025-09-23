import pandas as pd
import numpy as np

import os
import joblib

from datetime import date, datetime, timedelta
from time import strftime, time

import json

from pprint import pprint
import locale

from settings import DATA_DIR, START_DATE, DEBUG, logger
from settings import MODE, SAMPLE, DUCKDB, CHUNKSIZE, SAMPLE_SIZE
from settings import SELECTED_TICKERS, ticker_sets
from settings import MODELS, SELECTED_MODEL

from load_duckdb import duckdb_connect, db_table_records, db_get_ticker_records

# Use the system's default locale
locale.setlocale(locale.LC_ALL, '') 

# Pandas settings: Display all df columns
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 40)


# from scripts.transform import TransformData

# ML models and utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# import xgboost as xgb


# 2025 Update: top-190 US stocks with >$50b market cap
US_STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "WMT", "UNH", "V", "XOM", "MA", "PG", "JNJ", "COST", "ORCL", "HD", "ABBV",
    "BAC", "KO", "MRK", "NFLX", "CVX", "ADBE", "PEP", "CRM", "TMUS", "TMO", "AMD",
    "MCD", "CSCO", "WFC", "ABT", "PM", "DHR", "IBM", "TXN", "QCOM", "AXP", "VZ",
    "GE", "AMGN", "INTU", "NOW", "ISRG", "NEE", "CAT", "DIS", "RTX", "MS", "PFE",
    "SPGI", "UNP", "GS", "CMCSA", "AMAT", "UBER", "PGR", "T", "LOW", "SYK", "LMT",
    "HON", "TJX", "BLK", "ELV", "REGN", "BKNG", "COP", "VRTX", "NKE", "BSX", "PLD",
    "SCHW", "C", "PANW", "MMC", "ADP", "KKR", "UPS", "ADI", "AMT", "SBUX", "DE",
    "ANET", "BMY", "HCA", "CI", "KLAC", "FI", "LRCX", "BX", "GILD", "MU", "BA", "SO",
    "MDLZ", "ICE", "MO", "SHW", "DUK", "MCO", "CL", "INTC", "WM", "ZTS", "GD", "CTAS",
    "EQIX", "DELL", "NOC", "CME", "SCCO", "TDG", "SNYS", "APH", "WELL", "MCK", "PH",
    "PYPL", "ITW", "MSI", "PNC", "ABNB", "CMG", "USB", "CVS", "MMM", "FDX", "EOG",
    "ECL", "BDX", "CDNS", "TGT", "WDAY", "PLTR", "CSX", "ORLY", "CRWD", "MAR", "RSG",
    "AJG", "APO", "CARR", "EPD", "SPG", "APD", "AFL", "MRVL", "PSA", "DHI", "NEM",
    "FCX", "ROP", "SLB", "TFC", "FTNT", "EMR", "MPC", "NSC", "CEG", "PSX", "ADSK",
    "COF", "WMB", "ET", "IBKR", "GM", "MET", "O", "AEP", "OKE", "AZO", "HLT", "GEV",
    "SRE", "PCG", "DASH", "TRV", "CPRT", "OXY", "ROST", "KDP", "ALL", "BK", "DLR"
]

# https://companiesmarketcap.com/european-union/largest-companies-in-the-eu-by-market-cap/
EU_STOCKS = ['NVO','MC.PA', 'ASML', 'RMS.PA', 'OR.PA', 'SAP', 'ACN', 'TTE', 'SIE.DE','IDEXY','CDI.PA']
# https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/
INDIA_STOCKS = ['RELIANCE.NS','TCS.NS','HDB','BHARTIARTL.NS','IBN','SBIN.NS','LICI.NS','INFY','ITC.NS','HINDUNILVR.NS','LT.NS']


def save_dict(data, name, data_dir=DATA_DIR):
    file_name = f"{data_dir}{name}.json"
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent=4))

def load_dict(name, data_dir=DATA_DIR):
    file_name = f"{data_dir}{name}.json"
    with open(file_name) as f:
        data = json.loads(f.read())
    return data


class TransformData:
    tickers_df: pd.DataFrame
    transformed_df: pd.DataFrame

    def __init__(self, 
        # repo:DataRepository,
        tickers_df: pd.DataFrame,
        transformed_df: pd.DataFrame
        ):
        # copy initial dfs from repo
        # self.tickers_df = repo.ticker_df.copy(deep=True)
        # self.macro_df = repo.macro_df.copy(deep=True)
        # self.indexes_df = repo.indexes_df.copy(deep=True)
        
        self.tickers_df = tickers_df # .copy(deep=True)
        
        # init transformed_df
        self.transformed_df = None
        # self.transformed_df = transformed_df

    def transform(self):
        tickers = list(self.tickers_df.ticker.unique())
        tickers_df = self.tickers_df
        logger.debug(f" TransformData ({len(tickers)}): {tickers}")
        for ticker in tickers:
            logger.debug(f" ->> {ticker}")
            historyPrices = tickers_df[tickers_df["ticker"]==ticker].copy()
            historyPrices['Date'] = pd.to_datetime(historyPrices['date'])

            # generate features for historical prices, and what we want to predict

            if ticker in US_STOCKS:
                historyPrices['ticker_type'] = 'US'
            elif ticker in EU_STOCKS:
                historyPrices['ticker_type'] = 'EU'
            elif ticker in INDIA_STOCKS:
                historyPrices['ticker_type'] = 'INDIA'
            elif len(ticker)>6 and ticker[3]=='-':
                historyPrices['ticker_type'] = 'CRYPTO'
            else:
                historyPrices['ticker_type'] = 'ERROR'

            historyPrices['Ticker'] = ticker
            # historyPrices['Date'] = historyPrices.index.date
            historyPrices['Year']= historyPrices.Date.dt.year
            historyPrices['Month'] = historyPrices.Date.dt.month
            historyPrices['Weekday'] = historyPrices.Date.dt.weekday
            # print(historyPrices.head(2))

            # historical returns
            for i in [1,3,7,30,90,365]:
                historyPrices['growth_'+str(i)+'d'] = historyPrices['close'] / historyPrices['close'].shift(i)
            historyPrices['growth_future_30d'] = historyPrices['close'].shift(-30) / historyPrices['close']

            # Technical indicators
            # SimpleMovingAverage 10 days and 20 days
            historyPrices['SMA10']= historyPrices['close'].rolling(10).mean()
            historyPrices['SMA20']= historyPrices['close'].rolling(20).mean()
            historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
            historyPrices['high_minus_low_relative'] = (historyPrices['high'] - historyPrices['low']) / historyPrices['close']

            # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
            # Calculate daily returns first, then rolling std of returns
            historyPrices['daily_returns'] = historyPrices['close'].pct_change()
            historyPrices['volatility'] = historyPrices['daily_returns'].rolling(30).std() * np.sqrt(252)

            # what we want to predict (2025 update: changed from 5d to 30d)
            historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)

            if self.transformed_df is None:
                self.transformed_df = historyPrices
            else:
                self.transformed_df = pd.concat([self.transformed_df, historyPrices], ignore_index=True)


class TrainModel:
    transformed_df: pd.DataFrame #input dataframe from the Transformed piece 
    df_full: pd.DataFrame #full dataframe with DUMMIES

    # Dataframes for ML
    train_df:pd.DataFrame
    test_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_valid_df:pd.DataFrame

    X_train:pd.DataFrame
    X_valid:pd.DataFrame
    X_test:pd.DataFrame
    X_train_valid:pd.DataFrame
    X_all:pd.DataFrame

    # feature sets
    GROWTH: list
    OHLCV: list
    CATEGORICAL: list
    TO_PREDICT: list
    TECHNICAL_INDICATORS: list 
    TECHNICAL_PATTERNS: list
    MACRO: list
    NUMERICAL: list
    CUSTOM_NUMERICAL: list
    DUMMIES: list


    def __init__(self, transformed:TransformData):
        # init transformed_df
        self.transformed_df = transformed.transformed_df.copy(deep=True)
        self.transformed_df['ln_volume'] = self.transformed_df.volume.apply(lambda x: np.log(x) if x >0 else np.nan)
        # self.transformed_df['Date'] = pd.to_datetime(self.transformed_df['Date']).dt.strftime('%Y-%m-%d')

        self.models = {}
        self.ml_metrics = {}
        self.financial_metrics = {}
        self.threshold_analysis = {}
        
    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.transformed_df if (g.find('growth_')==0)&(g.find('future')<0)]
        self.OHLCV = ['open','high','low','close','volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
        self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future')>=0)]
        self.MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'DGS1', 'DGS5', 'DGS10']
        self.CUSTOM_NUMERICAL = ['vix_adj_close','SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']
        
        # artifacts from joins and/or unused original vars
        self.TO_DROP = ['Year','Date','Month_x', 'Month_y', 'index', 'Quarter','index_x','index_y'] + self.CATEGORICAL + self.OHLCV

        # All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
        self.TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
          'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
          'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
          'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
          'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
          'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
          'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
          'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
          'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
        self.TECHNICAL_PATTERNS =  [g for g in self.transformed_df.keys() if g.find('cdl')>=0]
        
        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + \
            self.CUSTOM_NUMERICAL + self.MACRO
        
        # CHECK: NO OTHER INDICATORS LEFT
        self.OTHER = [k for k in self.transformed_df.keys() if k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]
        return

    def _define_dummies(self):
        # dummy variables can't be generated from Date and numeric variables ==> convert to STRING (to define groups for Dummies)
        # self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.dt.strftime('%B')
        # self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.astype(str)

        # print(self.transformed_df.Month.unique())
        # print(self.transformed_df.Month.value_counts())
        # print(self.transformed_df[self.transformed_df.Month.isna()])
        self.transformed_df.loc[:,'Month'] = self.transformed_df.Month.astype(str)
        self.transformed_df['Weekday'] = self.transformed_df['Weekday'].astype(str)  

        # Generate dummy variables (no need for bool, let's have int32 instead)
        dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype='int32')
        self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
        # get dummies names in a list
        self.DUMMIES = dummy_variables.keys().to_list()

    def _perform_temporal_split(self, df:pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
        """
        Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

        Args:
            df (DataFrame): The DataFrame to split.
            min_date (str or Timestamp): Minimum date in the DataFrame.
            max_date (str or Timestamp): Maximum date in the DataFrame.
            train_prop (float): Proportion of data for training set (default: 0.7).
            val_prop (float): Proportion of data for validation set (default: 0.15).
            test_prop (float): Proportion of data for test set (default: 0.15).

        Returns:
            DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
        """
        # Define the date intervals
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

        logger.debug(f" Training data split dates: {min_date=:%x} {train_end=:%x} {val_end=:%x} {max_date=:%x}")

        # Assign split labels based on date ranges
        split_labels = []
        for date in df['Date']:
            if date <= train_end:
                split_labels.append('train')
            elif date <= val_end:
                split_labels.append('validation')
            else:
                split_labels.append('test')

        # Add 'split' column to the DataFrame
        df['split'] = split_labels

        return df


    def prepare_training_data(self):
        features_list = self.NUMERICAL+ self.DUMMIES
        # logger.debug(f"\nfeatures_list: {self.NUMERICAL} {self.DUMMIES} {features_list}")
        # logger.debug(f"\ncolumns: {self.df_full.columns}")
        features_list = [col for col in features_list if col in self.df_full.columns]
        # What we're trying to predict?
        to_predict = 'is_positive_growth_30d_future'

        # corr_is_positive_growth_5d_future = df_with_dummies[NUMERICAL+DUMMIES+TO_PREDICT].corr()['is_positive_growth_5d_future']
        corr_is_positive_growth_30d_future = self.df_full[features_list+[to_predict]].corr()[to_predict].sort_values(ascending=False)
        logger.debug(f"\n\ncorr_is_positive_growth_30d_future:\n{corr_is_positive_growth_30d_future}")

        self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
        self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train','validation'])].copy(deep=True)
        self.test_df =  self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)

        # Separate numerical features and target variable for training and testing sets
        self.X_train = self.train_df[features_list+[to_predict]]
        self.X_valid = self.valid_df[features_list+[to_predict]]
        self.X_train_valid = self.train_valid_df[features_list+[to_predict]]
        self.X_test = self.test_df[features_list+[to_predict]]
        # this to be used for predictions and join to the original dataframe new_df
        self.X_all =  self.df_full[features_list+[to_predict]].copy(deep=True)

        # Clean from +-inf and NaNs:

        self.X_train = self.clean_data(self.X_train)
        self.X_valid = self.clean_data(self.X_valid)
        self.X_train_valid = self.clean_data(self.X_train_valid)
        self.X_test = self.clean_data(self.X_test)
        self.X_all = self.clean_data(self.X_all)


        self.y_train = self.X_train[to_predict]
        self.y_valid = self.X_valid[to_predict]
        self.y_train_valid = self.X_train_valid[to_predict]
        self.y_test = self.X_test[to_predict]
        self.y_all =  self.X_all[to_predict]

        # remove y_train, y_test from X_ dataframes
        del self.X_train[to_predict]
        del self.X_valid[to_predict]
        del self.X_train_valid[to_predict]
        del self.X_test[to_predict]
        del self.X_all[to_predict]

        logger.debug(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
        logger.debug(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')     

    def clean_data(self, df:pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df   

    def prepare_dataframe(self):
        logger.debug("  --> Prepare the dataframe: define feature sets, add dummies, temporal split")
        self._define_feature_sets()
        # get dummies and df_full
        self._define_dummies()
        
        logger.debug(self.df_full.head(5).T) # TEMP

        # temporal split
        min_date_df = self.df_full.Date.min()
        max_date_df = self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        # define dataframes for ML
        self.prepare_training_data()

        return
      
    def train_randomforest(self, max_depth=17, n_estimators=200):
        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        logger.debug(f'Training the best model (RandomForest (max_depth={max_depth}, n_estimators={n_estimators}))')
        self.model = RandomForestClassifier(n_estimators = n_estimators,
                                     max_depth = max_depth,
                                     random_state = 42,
                                     n_jobs = -1)

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def train_decisiontree(self, max_depth=10, min_samples_leaf=3):
        # https://scikit-learn.org/stable/modules/tree.html
        logger.debug(f'Training the best model (DecisionTree (max_depth={max_depth}, min_samples_leaf={min_samples_leaf}))')
        self.model = DecisionTreeClassifier(max_depth = max_depth,
                                     min_samples_leaf = min_samples_leaf,
                                     random_state = 42,)

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def train_adaboost(self, learning_rate=1, n_estimators=100):
        # https://scikit-learn.org/stable/modules/ensemble.html#adaboost
        logger.debug(f'Training the best model (AdaBoost (learning_rate={learning_rate}, n_estimators={n_estimators}))')
        self.model = AdaBoostClassifier(algorithm='SAMME', 
                                     n_estimators = n_estimators,
                                     learning_rate = learning_rate,
                                     random_state = 42,
                                     )

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def train_xgboost(self, learning_rate=1, n_estimators=100):
        # https://scikit-learn.org/stable/modules/ensemble.html#xgboost
        logger.debug(f'Training the best model (AdaBoost (learning_rate={learning_rate}, n_estimators={n_estimators}))')
        self.model = AdaBoostClassifier(algorithm='SAMME', 
                                     n_estimators = n_estimators,
                                     learning_rate = learning_rate,
                                     random_state = 42,
                                     )

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def save_model(self, data_dir:str, model_name:str):
        '''Save model to file in a local directory 'dir' '''
        os.makedirs(data_dir, exist_ok=True)      

        # Save the model to a file
        model_filename = f'{model_name}_model.joblib'
        path = os.path.join(data_dir,model_filename)
        joblib.dump(self.model, path)

    def load_model(self, data_dir:str, model_name:str):
        """Load files from the local directory"""
        os.makedirs(data_dir, exist_ok=True)   
        # Load the model from a file
        model_filename = f'{model_name}_model.joblib'
        path = os.path.join(data_dir,model_filename)
        self.model  = joblib.load(path)

    def make_prediction(self, pred_name:str):
        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        logger.debug('  --> Making prediction: {pred_name}')
        
        y_pred_all = self.model.predict_proba(self.X_all)
        y_pred_all_class1 = [k[1] for k in y_pred_all] #list of predictions for class "1"
        y_pred_all_class1_array = np.array(y_pred_all_class1) # (Numpy Array) np.array of predictions for class "1" , converted from a list

        self.df_full[pred_name] = y_pred_all_class1_array
        
        # define rank of the prediction
        self.df_full[f"{pred_name}_rank"] = self.df_full.groupby("Date")[pred_name].rank(method="first", ascending=False)

    def evaluate_fin_metrics(self,
                            predictions_df: pd.DataFrame,
                            model_name: str,
                            y_proba_col: str,
                            investment_per_signal: float = 100,
                            transaction_cost: float = 0.001,
                            thresholds: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]) -> dict[str, float]:
        logger.debug(f'  --> Evaluating fin metrics: {model_name} {y_proba_col}')
        # y_proba_col = f'y_proba_{model_name}'
        if y_proba_col not in predictions_df.columns:
            logger.error(f"Probability column {y_proba_col} not found")
            return {}
        
        financial_results = []
        
        for threshold in thresholds:
            # Generate signals following original notebook approach
            signals = (predictions_df[y_proba_col] >= threshold).astype(int)
            trades_df = predictions_df[signals == 1].copy()
            
            if len(trades_df) == 0:
                # No trades generated
                financial_results.append({
                    'threshold': threshold,
                    'total_trades': 0,
                    'cagr': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'hit_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_return': 0.0,
                    'avg_trade_return': 0.0
                })
                continue
            
            # Calculate trade P&L following original notebook approach:
            # sim1_gross_rev = prediction * 100 * (growth_future_30d - 1)
            # sim1_fees = -prediction * 100 * 0.002  # 0.2% total transaction cost
            # sim1_net_rev = gross_rev + fees
            
            trades_df['gross_pnl'] = investment_per_signal * (trades_df['growth_future_30d'] - 1)
            trades_df['transaction_costs'] = -investment_per_signal * (transaction_cost * 2)  # Buy + sell
            trades_df['net_pnl'] = trades_df['gross_pnl'] + trades_df['transaction_costs']
            trades_df['trade_success'] = (trades_df['growth_future_30d'] > 1.0).astype(int)
            
            # Aggregate daily P&L
            daily_pnl = trades_df.groupby(trades_df.index)['net_pnl'].sum().sort_index()
            
            # Calculate performance metrics following original notebook approach
            total_trades = len(trades_df)
            total_net_pnl = trades_df['net_pnl'].sum()
            total_gross_pnl = trades_df['gross_pnl'].sum()
            hit_rate = trades_df['trade_success'].mean()
            
            # CAGR calculation (following original notebook)
            # Estimate required capital based on daily investment patterns
            daily_trades = trades_df.groupby(trades_df.index).size()
            avg_daily_trades = daily_trades.mean()
            q75_daily_trades = daily_trades.quantile(0.75)
            estimated_capital = investment_per_signal * 30 * q75_daily_trades  # 30 days holding period
            
            if estimated_capital > 0:
                total_return = total_net_pnl / estimated_capital
                
                # Time period calculation
                start_date = trades_df['Date'].min()
                end_date = trades_df['Date'].max()
                years = (end_date - start_date).days / 365.25
                logger.debug(f"cagr {years=} {start_date=} {end_date=}")
                
                if years > 0:
                    cagr = ((estimated_capital + total_net_pnl) / estimated_capital) ** (1/years) - 1
                else:
                    cagr = 0.0
            else:
                total_return = 0.0
                cagr = 0.0
            
            # Sharpe ratio calculation
            if len(daily_pnl) > 1:
                daily_returns = daily_pnl / estimated_capital if estimated_capital > 0 else daily_pnl / investment_per_signal
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown calculation
            cumulative_pnl = daily_pnl.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / estimated_capital if estimated_capital > 0 else (cumulative_pnl - running_max)
            max_drawdown = drawdown.min()
            
            # Profit factor
            winning_trades = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
            losing_trades = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else np.inf if winning_trades > 0 else 0
            
            financial_results.append({
                'threshold': threshold,
                'total_trades': total_trades,
                'cagr': cagr * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,  # Convert to percentage
                'hit_rate': hit_rate * 100,  # Convert to percentage
                'profit_factor': profit_factor,
                'total_return': total_return * 100,  # Convert to percentage
                'avg_trade_return': trades_df['net_pnl'].mean(),
                'estimated_capital': estimated_capital,
                'total_net_pnl': total_net_pnl,
                'avg_daily_trades': avg_daily_trades
            })
        
        # Store results
        financial_metrics = {
            'model_name': model_name,
            'threshold_analysis': financial_results
        }
        
        self.financial_metrics[model_name] = financial_metrics
        
        # Log best performance
        best_by_sharpe = max(financial_results, key=lambda x: x['sharpe_ratio'])
        logger.info(f"Fin metrics evaluated for {model_name}: Best Sharpe={best_by_sharpe['sharpe_ratio']:.2f} at threshold={best_by_sharpe['threshold']}")
        
        try:
            save_dict(financial_metrics, model_name, DATA_DIR)
        except Exception as e:
            logger.error(f"!! Error saving {financial_metrics}: {e}")

        return financial_metrics

def train_model(df_tickers_prices, model_name=SELECTED_MODEL):
    logger.info(f"\n --> Training model {model_name}")

    if len(df_tickers_prices):
        transformed = TransformData(df_tickers_prices, None)
        transformed.transform()
        logger.debug(f"  --> Enriched tickers data: {len(transformed.transformed_df)}")

        model_trainer = TrainModel(transformed)
        print(model_trainer.transformed_df.columns)
        # print(model_trainer.transformed_df.head(5).to_string())
        # print('\nticker.unique', model_trainer.transformed_df.ticker.unique())
        # print('\nMonth.unique', model_trainer.transformed_df.Month.unique())
        model_trainer.prepare_dataframe()
        if model_name=="RandomForest":
            model_trainer.train_randomforest()
        elif model_name=="DecisionTree":
            model_trainer.train_decisiontree()
        elif model_name=="adaboost":
            model_trainer.train_adaboost()
        # elif model_name=="xgboost":
        #     model_trainer.train_xgboost()
        else:
            logger.error(f"!! model not supported: {model_name}")
            return 1


        # print(f"\n\ndf_full:")
        # print(model_trainer.df_full.groupby(['ticker'])['Date'].agg(['min','max','count']))
        model_trainer.save_model(DATA_DIR, model_name)
        # model_trainer.load_model(DATA_DIR, model_name)

        prediction_name='pred_rf_30d_best'
        model_trainer.make_prediction(pred_name=prediction_name)
        logger.debug(f"  --> Trained model data: {len(model_trainer.df_full)}")

        df_simulation = model_trainer.df_full
        fin_metrics = model_trainer.evaluate_fin_metrics(
                                  predictions_df=df_simulation,
                                  model_name=model_name,
                                  y_proba_col=prediction_name,
                                )
        logger.debug(f"  --> fin metrics: {fin_metrics}")

    return 0

def train_models(mode=DUCKDB, models=MODELS): ## MODELS [SELECTED_MODEL]
    if mode == DUCKDB:
        # duckdb_transform_data(table_name="tickers_data", description="Enriching data")
        con, tables = duckdb_connect()
        df_tickers_prices = db_table_records(con, table_name="tickers_prices")
        logger.debug(f"  --> Load tickers_prices: {len(df_tickers_prices)}")
    # elif mode == BIGQUERY: # TODO

    if len(df_tickers_prices)==0:
        logger.error(f"!! df_tickers_prices empty! Stopping.")
        return 1

    for model_name in models:
        res = train_model(df_tickers_prices, model_name)
        if res==1:
            return 1
    # all done
    return 0

if __name__ == "__main__":
    from load_duckdb import db_columns
    
    selected_tickers = SELECTED_TICKERS # ticker_sets["set1"]
    # table_name = "tickers_prices"  # 'tickers_info' 'tickers_prices'
    if False and DEBUG:
        con, tables = duckdb_connect()
        print(tables)
        db_columns(con, table_name='tickers_data', with_sample=True) # transformed
        df_tickers_prices = db_table_records(con, table_name="tickers_prices")
        print(df_tickers_prices.head())
        # df_tickers_data = db_get_ticker_records(con, table_name='tickers_data', ticker="AAPL", start_date=START_DATE, end_date=None)
        # print(df_tickers_data.head(5).T)
        # print(df_tickers_data["ticker"].unique())
        # exit()


        transformed = TransformData(df_tickers_prices, None)
        transformed.transform()

        model_trainer = TrainModel(transformed)
        print(model_trainer.transformed_df.columns)
        # print(model_trainer.transformed_df.head(5).to_string())
        # print('\nticker.unique', model_trainer.transformed_df.ticker.unique())
        # print('\nMonth.unique', model_trainer.transformed_df.Month.unique())
        model_trainer.prepare_dataframe()

        model_name = SELECTED_MODEL
        if model_name=="RandomForest":
            model_trainer.train_randomforest()
        elif model_name=="DecisionTree":
            model_trainer.train_decisiontree()
        elif model_name=="adaboost":
            model_trainer.train_adaboost()
        # print(f"\n\ndf_full:")
        # print(model_trainer.df_full.groupby(['ticker'])['Date'].agg(['min','max','count']))

        model_trainer.save_model(DATA_DIR, model_name)
        model_trainer.load_model(DATA_DIR, model_name)

        prediction_name='pred_rf_30d_best'
        model_trainer.make_prediction(pred_name=prediction_name)
        COLUMNS = ['close','ticker','Date', prediction_name, prediction_name+'_rank']

        print('Results of the 30-day prediction estimation (last 10 days):')

        print('Top 3 predictions every day (30-day future growth):')
        print(model_trainer.df_full[model_trainer.df_full[f'{prediction_name}_rank']<=3].sort_values(by=["Date",f'{prediction_name}_rank']).tail(20)[COLUMNS])

        print('Bottom 3 predictions every day (30-day future growth):')
        max_date = model_trainer.df_full.Date.max()
        count_predictions = model_trainer.df_full[model_trainer.df_full.Date==max_date].Ticker.nunique()
        print(model_trainer.df_full[model_trainer.df_full[f'{prediction_name}_rank']>=count_predictions-2].sort_values(by=["Date",f'{prediction_name}_rank']).tail(20)[COLUMNS])

        # simulation_date = pd.to_datetime(date(max_date.year, 1, 1)) # max_date + pd.Timedelta(days=-60)
        simulation_date = max_date + pd.Timedelta(days=-int(366*1.5))
        print(f"\n{simulation_date=:%x} {max_date=:%x}")
        df_simulation = model_trainer.df_full[model_trainer.df_full['Date']>=simulation_date]
        fin_metrics = model_trainer.evaluate_fin_metrics(
                                  predictions_df=df_simulation,
                                  model_name="RandomForest",
                                  y_proba_col="pred_rf_30d_best",
                                )

        print(f"fin_metrics {fin_metrics.get("model_name", '')}:")
        for threshold in fin_metrics.get("threshold_analysis", []):
            pprint(f" {threshold}")
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"Current date and time: {current_datetime}")
        exit()

    train_models(mode=MODE)
