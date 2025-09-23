import requests
from io import StringIO
import pandas as pd

from datetime import date, datetime, timedelta
from time import sleep, strftime, time

import yfinance as yf
# import logging

from settings import DATA_DIR, SCRAPE, START_YEAR, START_DATE, DEBUG, logger
from settings import MODE, SAMPLE, DUCKDB, CHUNKSIZE, SAMPLE_SIZE
from settings import SELECTED_TICKERS, ticker_sets

# Pandas settings: Display all df columns
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 40)


def normalize_df_columns(df):
    # normalizing column names - lowercase, no spaces
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


# extract companies & tickers list by scraping of S&P 500 data from Wikipedia
def extract_snp500_list(scrape=SCRAPE):
    if scrape:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }        
        response = requests.get(url, headers=headers)
        df_list = pd.read_html(StringIO(response.text), header=0)
        # we need the first table from the webpage - df_list[0]
        df = df_list[0]
        df = normalize_df_columns(df)
        # renaming columns
        df.columns = df.columns.str.replace("symbol", "ticker")
        df.columns = df.columns.str.replace("security", "company")
        df.columns = df.columns.str.replace("gics_sector", "sector")
        # fixing 'BRK.B' to 'BRK-B' etc
        df["ticker"] = df["ticker"].str.replace(".", "-")
        # fixing founded: like '2013 (1888)' -> int year
        df["founded"] = df["founded"].astype(str).str[0:4].astype(int)
        df.columns = df.columns.str.replace("founded", "founded_year")
        # location: only last part "North Chicago, Illinois" -> "Illinois"
        df["location"] = df["headquarters_location"].map(lambda col: col.split(", ")[-1])

        df.to_csv(DATA_DIR + f"snp500.csv", encoding="utf-8", index=False)
        if DEBUG:
            # save historical data to reuse without scraping again
            df.to_csv(DATA_DIR + f"snp500_{strftime('%Y-%m-%d')}.csv", encoding="utf-8", index=False)
    else:
        df = pd.read_csv(DATA_DIR + f"snp500.csv", encoding="utf-8")
        df["founded_year"] = df["founded_year"].astype(int)

    # we need only few columns
    selected_columns = ["ticker", "company", "sector", "location", "founded_year", "date_added"]
    return df[selected_columns]


def filter_df(df, tickers, start_year=START_YEAR):
    try:
        df1 = df[(df["ticker"].isin(tickers)) & (df["date"].dt.year >= int(start_year))]
    except:
        df1 = df[(df["ticker"].isin(tickers)) & (df["date"] >= str(start_year))]
    return df1


def calculate_tickers_data(df):
    # Calculate new features for each ticker
    # https://www.investopedia.com/terms/m/movingaverage.asp
    # https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
    # ! df should be for 1 ticker !
    df["MA_10"] = df["Close"].rolling(window=10).mean()  # 10-day moving average
    df["MA_20"] = df["Close"].rolling(window=20).mean()  # 20-day moving average
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()  # 20-day exponential moving average
    df["Momentum"] = df["Close"] - df["Close"].shift(5)  # Momentum
    df["Volatility"] = df["Close"].rolling(window=20).std()  # 20-day rolling volatility
    return df


def extract_tickers_data(tickers, start_date=START_DATE, end_date=None, scrape=True):
    logger.debug(f"extract_tickers_data: {tickers}, {start_date}, {end_date}, {scrape}")
    
    # yf_logger = yf.utils.get_yf_logger()
    # # Set the logging level (e.g., to CRITICAL to suppress most messages)
    # if DEBUG:
    #     pass # let yf print progress messages
    # else: # prevent verbose logging
    #     yf_logger.setLevel(logging.CRITICAL)
    #     yf_logger.disabled = True 

    t_start = time()
    if scrape:
        if end_date == None:
            end_date = strftime("%Y-%m-%d")  # date.today().strftime("%Y-%m-%d")
        tickers_data = []
        failed_tickers = []
        for ticker in tickers:
            # Fetch the data
            data = yf.download(ticker, group_by="Ticker", start=start_date, end=end_date, auto_adjust=True, progress=DEBUG)
            df_ = data.stack(level=0, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index(level=1)
            logger.debug(f" Downloaded {ticker}: {df_.shape[0]} record(s)")
            if df_.shape[0] == 0:
                failed_tickers.append(ticker)
            sleep(0.7)  # delay to prevent yf scraping errors "rate limit exceeded"
            # extra calculations
            df_ = calculate_tickers_data(df_)
            tickers_data.append(df_)
        df = pd.concat(tickers_data)
        df.reset_index(inplace=True)

        if failed_tickers:
            logger.error(f"\n!! failed_tickers: {failed_tickers}\n")
        else:
            logger.debug("All tickers downloaded normally.\n")

        # extra calculations
        df["Next_Day_Return"] = df.groupby("Ticker")["Close"].shift(-1) / df["Close"] - 1

        # date to short string
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        df = normalize_df_columns(df)  # lower case, no spaces

        # !! reasonable presision in CSV: float_format='%g' for smaller file size
        # With no precision given, uses a precision of 6 significant digits for float
        df.to_csv(DATA_DIR + f"tickers_prices.csv", encoding="utf-8", index=False, float_format="%g")
        # df.to_csv(DATA_DIR+f"tickers_{start_date}-{end_date}.csv", encoding='utf-8', index=False)
        records_read = df.shape[0]
    else:
        try:
            if start_date and end_date:
                df = pd.read_csv(DATA_DIR + f"tickers_{start_date}-{end_date}.csv")  # , parse_dates=[1], date_format="ISO8601")
            else:
                df = pd.read_csv(DATA_DIR + f"tickers_prices.csv", encoding="utf-8")  # , parse_dates=[1], date_format="ISO8601")
        except:
            df = pd.read_csv(DATA_DIR + f"tickers_prices.csv", encoding="utf-8")  # , parse_dates=[1], date_format="ISO8601")

        records_read = df.shape[0]
        # filtering read data with selected tickers and start_date
        df = filter_df(df, tickers, start_year=start_date[0:4])

    # Date to normal date format
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Total selected: {df.shape[0]}/{records_read} record(s), took {(time() - t_start):.3f} second(s)")
    return df


def split_dataframe(df, chunk_size=CHUNKSIZE):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


def df_latest_date(df, col="date"):
    try:
        d = df[col].max().strftime("%Y-%m-%d")
    except:
        d = None
    return d


def extract_data(mode=SAMPLE, selected_tickers=SELECTED_TICKERS, scrape=SCRAPE):
    logger.info(f"\n -> Extract data: mode {mode}")

    logger.debug(f'\n[{strftime("%H:%M:%S")}] Extracting data: mode {mode}')

    # part 1: tickers_info
    df_tickers_info = extract_snp500_list(scrape=scrape)  # True False

    # part 2: tickers_prices
    if len(selected_tickers) == 0:
        # on empty list use all S&P500 tickers
        selected_tickers = df_tickers_info["ticker"].to_list()

    # scrape = False # False True
    process = "Scraping" if scrape else "Loading"
    logger.debug(f"\n{process} prices. Selected Tickers ({len(selected_tickers)}): {selected_tickers}\n")

    # table_name = "tickers_prices"
    # export_filename = DATA_DIR + table_name
    yesterday = date.today() - timedelta(days=1)  # TEMP
    start_date = START_DATE
    end_date = yesterday  # Until market is closed we have no data for the current date
    df_tickers = extract_tickers_data(selected_tickers, start_date, end_date, scrape=scrape)  # True False
    if "adj_close" in df_tickers.columns:
        # since yfinance 0.2.51 Close price is already adjusted by default
        # remove for old scraped CSVs
        df_tickers.drop(["adj_close"], axis=1, inplace=True)

    logger.debug(" Latest date: {df_latest_date(df_tickers)}")

    return df_tickers_info, df_tickers


def df_tickers_info_eda(df_tickers_info):
    print("\nS&P 500\n")
    print(df_tickers_info.info())
    print(df_tickers_info.head(30))
    print("\n", df_tickers_info["sector"].value_counts())
    print("\n", df_tickers_info["location"].value_counts().head(10))
    print("\nfounded\n", df_tickers_info["founded_year"].value_counts(bins=10))
    print("\ndate_added\n", df_tickers_info["date_added"].str[0:4].astype(int).value_counts(bins=10))
    year_founded = 2020
    df2020 = df_tickers_info[df_tickers_info["founded_year"] >= year_founded]
    print(f"\nfounded {year_founded}+ ({df2020.shape[0]}):\n", df2020.head(30))


def df_tickers_eda(df_tickers_info, df_tickers):
    print(df_tickers.info())
    print(df_tickers.head(5))
    print("\n", df_tickers["ticker"].value_counts().head(30))
    start_date = datetime(2024, 10, 1)
    df_tickers1 = df_tickers[df_tickers["date"] >= start_date]
    df_tickers1["week_number"] = df_tickers1["date"].dt.isocalendar().week
    print(f"\nTickers {start_date}+ ({df_tickers1.shape[0]}):\n", df_tickers1.head(30))

    df_agg = (
        df_tickers1.groupby(
            [
                "ticker",
                # 'week_number',
            ]
        )
        .agg({"low": "mean", "high": "mean", "close": "mean"})
        .reset_index()
    )  # 'mean', 'sum', 'count'
    # , 'adj_close':'mean'
    print(f"\nTickers agg {start_date}+ ({df_agg.shape[0]}):\n", df_agg.head(30))
    print(f"\nTickers agg {start_date}+ ({df_agg.shape[0]}):\n", df_agg["close"].value_counts(bins=10))

    #
    df2 = pd.merge(df_agg, df_tickers_info, left_on="ticker", right_on="ticker", how="left")
    print(f"\nTickers agg {start_date}+ ({df2.shape[0]}):\n", df2["sector"].value_counts())


##################
if __name__ == "__main__":
    selected_tickers = ticker_sets["set1"]

    # df_tickers_info, df_tickers = extract_data(mode=DUCKDB, selected_tickers=selected_tickers, scrape=True)
    df_tickers_info, df_tickers = extract_data(mode=SAMPLE, selected_tickers=selected_tickers, scrape=False)  # True False

    # some quick tests/exploration/analytics
    # df_tickers_info_eda(df_tickers_info)
    # df_tickers_eda(df_tickers_info, df_tickers)

    # build_portfolio
    # optimal_portfolio(df_tickers, selected_tickers, budget=25000, start_date='2010-01-01', end_date='2025-03-31', verbose=True)
