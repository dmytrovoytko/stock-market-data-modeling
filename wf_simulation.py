import pandas as pd

from datetime import date, datetime, timedelta
from time import strftime, time

from settings import DATA_DIR, START_DATE, DEBUG, logger
from settings import MODE, SAMPLE, DUCKDB, CHUNKSIZE, SAMPLE_SIZE
from settings import SELECTED_TICKERS, ticker_sets

# Pandas settings: Display all df columns
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 40)


def simulate(mode=DUCKDB):
    logger.info(f"\n --> Simulate: mode {mode}")
    if mode == DUCKDB:
        # duckdb_transform_data(table_name="tickers_data", description="Enriching data")
        pass
    # elif mode == BIGQUERY: # TODO

    return 0

if __name__ == "__main__":
    selected_tickers = ticker_sets["set1"]

