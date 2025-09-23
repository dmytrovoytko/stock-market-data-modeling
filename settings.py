import os, sys, logging
from dotenv import load_dotenv
from time import strftime

# loading environment variables
load_dotenv()

# show extra information for checking execution
DEBUG = True # True # False
DEBUG_LOGS = False # True # False

DATA_DIR = f'./data/'  # ! with '/' at the end!
VISUALS_DIR = './visuals/'

SCRAPE = False # True # False 
START_YEAR = 2015 # 2010
START_DATE = f'{START_YEAR}-01-01'


def get_custom_logger(name):
    # Create a logger with name
    logger = logging.getLogger(name)
    logger.setLevel((logging.DEBUG if DEBUG else logging.INFO))

    # Set formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s') # compact

    if DEBUG and DEBUG_LOGS:
        # Create a FileHandler for logging to a file
        cur_date = strftime('%Y-%m-%d %H%M')
        log_name = DATA_DIR + cur_date + ".log"
        file_handler = logging.FileHandler(log_name)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Create a StreamHandler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel((logging.DEBUG if DEBUG else logging.INFO))
    logger.addHandler(stream_handler)

    return logger

# this logger will be used in all modules
logger = get_custom_logger('SMA proj')


def load_duckdb_settings():
    # using environment variables, including DUCKDB connection settings
    connection = os.environ.get('DUCKDB_CONNECTION', None)
    if connection:
        logger.debug(f'DUCKDB_CONNECTION: {connection}')
        return connection
    else:
        logger.warning('The DUCKDB_CONNECTION environment variable is not defined.\n')
        return None


# define your list like below
tickers_my = [
            "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", # , "GOOGL"
            "AMD", "INTC", "BRK-B", 
            # "ADBE", "AVGO", "CRWD", "CSCO", "DELL", "FDX", "HPE", "HPQ", "IBM", 
            # "MU", "MSI", "ORCL", "PLTR", "QCOM", "CRM", "SMCI", "WDC",
            "PLTR",
            # indexes
            # "^SPX", "^SP400", "^SP600", "^NY", "^DJUS", "^NYA", "^XAX", "^IXIC",
            "^SPX", "^DJUS", "^IXIC",
            # 
            "VOO", "SPY", 
            # crypto
            "BTC-USD", "ETH-USD", "SOL-USD", 
            ]

tickers_indexes = [ 
            # "^GSPC", ==  "^SPX" 
            "^SPX", "^SP400", "^SP600", "^NY", "^DJUS", "^NYA", "^XAX", "^IXIC",
            "^NDX", "^NDXE", "^NXTQ", "^RUI", "^RUT", "^RUA", 
            "^FTEU1", "^SPEUP", "^N100", "^ISEQ",
            # The ticker for the S&P 500 index is ^GSPC, but it cannot be traded. 
            # SPX and SPY represent options on the S&P 500 index, and they are traded in the market.
            # "VOO", "SWPPX", "IVV", "VFIAX", "SPY", "", 
            ]

tickers_crypto = [
            "BTC-USD", "ETH-USD", "USDT-USD", "XRP-USD", "BNB-USD", "SOL-USD", "USDC-USD", "LTC-USD",
            ]

ticker_sets = {
    # "all": [], # to extract all S&P500 tickers
    "magnificent": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", ],
    # "crypto": tickers_crypto,
    # "indexes": tickers_indexes,
    "set1": tickers_my,
} 
SELECTED_TICKERS = ticker_sets["magnificent"]                    
# SELECTED_TICKERS = ticker_sets["set1"]                    

MODELS = [
    "RandomForest",
    "DecisionTree",
    "adaboost",
    # "xgboost",
]

SELECTED_MODEL = "RandomForest"
SELECTED_MODEL = "DecisionTree"
SELECTED_MODEL = "adaboost"

SAMPLE = 'sample' # export samples to .csv 
PARQUET = 'parquet' # export to .parquet files
DUCKDB = 'duckdb' # export to DuckDB

MODE = os.environ.get('DATAWAREHOUSE', DUCKDB) # DUCKDB SAMPLE
CHUNKSIZE = 100_000 # 100_000
SAMPLE_SIZE = 1000


if MODE==DUCKDB:
    DUCKDB_CONNECTION = load_duckdb_settings()
else:
    DUCKDB_CONNECTION = None
