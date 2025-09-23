import os
from glob import glob
from time import strftime, time

import duckdb

from settings import DATA_DIR, VISUALS_DIR, DEBUG, logger
from settings import START_YEAR, START_DATE
from settings import DUCKDB_CONNECTION


def db_tables(con, table_names=""):
    # table_names = "'tickers_info', 'tickers_prices'"
    if table_names:
        filters = f"WHERE table_name in ({table_names}) "
    else:
        filters = ""
    res = con.sql(f"SELECT * FROM duckdb_tables() " + filters)
    print("\ndb_tables:", table_names)
    logger.debug(res)
    ddf = res.df()
    print(ddf["table_name"].to_list())  # , ddf['sql'].to_list()


def db_columns(con, table_name, with_sample=False):
    print(f"\ndb_columns {table_name}:")
    res = con.sql(
        f"SELECT * FROM duckdb_columns() \
                    WHERE table_name ILIKE '{table_name}' "
    )
    # print(res)
    df = res.df()
    print(df[["column_name", "data_type"]].to_string())  # , 'character_maximum_length', 'numeric_precision']])

    if with_sample:
        res = con.sql(f"SELECT * FROM {table_name} LIMIT 5")
        print(f"Head:\n{res}")  # .fetchall()

    res = con.sql(f"SELECT count(*) FROM {table_name}")
    print(" Total records:", res.fetchone()[0])


def db_table_records(con, table_name):
    res = con.sql(f"SELECT * FROM {table_name}")
    df = res.df()
    return df

def db_table_recnumber(con, table_name):
    res = con.sql(f"SELECT count(*) FROM {table_name}")
    recnumber = res.fetchone()[0]
    print(" Total records:", recnumber)
    return recnumber


def db_get_ticker_records(con, table_name, ticker, start_date=START_DATE, end_date=None):
    # momentum, volatility FROM {table_name} \
    query = f"SELECT date, open, high, low, close, volume, \
                ma_7, ma_10, ma_20, ma_30, ma_100, macd, rsi FROM  {table_name} \
                WHERE ticker='{ticker}' \
                    AND strftime(date, '%Y-%m-%d') >= '{start_date}' \
                ORDER BY date ASC;" # date_part('year', date)
    res = con.sql(query)
    df = res.df()
    logger.debug(" Total records: {df.shape[0]}")
    return df


def duckdb_connect(connection=""):
    if connection == "":
        if DUCKDB_CONNECTION:
            connection = DUCKDB_CONNECTION
        else:
            logger.error("DUCKDB_CONNECTION environment variable is not defined.\n")
            return None, []

    if not connection.startswith("md:") and not os.path.exists(connection):
        logger.warning(f'The file "{connection}" does not exist. Creating.\n')
    elif connection.startswith("md:") and not os.environ.get("motherduck_token", ""):
        logger.error(f'!! Error connecting to MotherDuck "{connection}": motherduck_token not defined. Check .env file.')
        return None, []

    logger.debug(f'\nDuckDb connection: "{connection}"')

    try:
        con = duckdb.connect(connection, read_only=False)
        df = con.sql(f"SELECT * FROM duckdb_tables()").df()
        tables = df["table_name"].to_list()
        return con, tables
    except Exception as e:
        logger.error(f'!! Error connecting to duckdb "{connection}":\n {e}')
        return None, []


def duckdb_load_data(df, table_name, description, connection=""):
    logger.debug(f"\n[{strftime("%H:%M:%S")}] Loading: {description} {table_name} {connection}")
    con, tables = duckdb_connect(connection)
    if not con:
        return 1

    t_start = time()
    try:
        if table_name in tables:
            logger.debug(f"\nTable {table_name} exists -> INSERT OR REPLACE INTO...")
            # con.sql(f'INSERT OR IGNORE INTO {table_name} SELECT * FROM df')
            con.sql(f"INSERT OR REPLACE INTO {table_name} SELECT * FROM df")
        else:
            logger.debug(f"\nCreating table {table_name}...")
            con.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            # to prevent duplicated records we need to add PRIMARY KEY
            if table_name == "tickers_prices":
                # composite PRIMARY KEY
                con.sql(f"ALTER TABLE {table_name} ADD PRIMARY KEY (date, ticker);")
            elif table_name == "tickers_info":
                con.sql(f"ALTER TABLE {table_name} ADD PRIMARY KEY (ticker);")

        res = con.execute(f"SELECT COUNT(*) FROM {table_name}")
        logger.debug(f" Loaded. Total records in {table_name}: {res.fetchone()[0]}, took {(time() - t_start):.3f} second(s)")
    except Exception as e:
        logger.error(f"! Error while loading {description} to duckdb {table_name}.\n{e}")
        return 1

    # print(f'Table description: {table.description}')
    # print(f'Table schema: {table.schema}')

    return 0


def duckdb_transform_data(table_name, year=START_YEAR, description="", connection=""):
    logger.debug(f"\n[{strftime("%H:%M:%S")}] Transforming: {description} {table_name} {connection}")
    con, tables = duckdb_connect(connection)
    if not con:
        return 1

    logger.debug(f"\ntransform_data in {table_name}:")

    # Adding tickers_info to prices
    res = con.sql(
        f"CREATE OR REPLACE TABLE {table_name} AS "
        + f"""( 
        WITH prices AS (
          SELECT 
            date, ticker, open, high, low, close, volume,
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7,
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS ma_10,
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20,
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30,
            AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS ma_100,
            SUM(CASE WHEN close > open THEN close - open ELSE 0 END) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS gain,
            SUM(CASE WHEN open > close THEN open - close ELSE NULL END) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS loss,
            MAX(high) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS highest,
            MIN(low) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS lowest,
            (AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) - AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)) AS macd
          FROM tickers_prices
        )
          SELECT 
            date, prices.ticker, open, high, low, close, volume,
            ma_7, ma_10, ma_20, ma_30, ma_100, macd, 
            (100 - (100 / (1 + (gain / loss)))) AS rsi,
            info.sector as sector
          FROM prices
          LEFT JOIN tickers_info AS info
                    ON prices.ticker = info.ticker
        );""")

    # df = res.df()
    # print(df.shape[0])
    # print(df.head(5))
    # print(res)

    # If tickers are not in S&P500, try to set sector 'Market Indexes'
    res = con.sql(
        f"UPDATE {table_name} \
                        SET sector = 'Market Indexes' \
                        WHERE sector IS NULL and left(ticker, 1)='^';"
    )
    # df = res.df()
    # print(df.shape[0])
    # print(df.head(5))
    # print(res)

    # ETF
    res = con.sql(
        f"UPDATE {table_name} \
                        SET sector = 'ETF' \
                        WHERE sector IS NULL and ticker in ('VOO','SPY');"
    )

    # crypto
    res = con.sql(
        f"UPDATE {table_name} \
                        SET sector = 'Crypto' \
                        WHERE sector IS NULL and right(ticker, 4)='-USD';"
    )

    # Check If tickers have empty sector
    res = con.sql(
        f"SELECT * FROM {table_name} \
                        WHERE sector IS NULL or trim(sector)='';"
    )
    df = res.df()
    if df.shape[0]:
        print("\nEmpty sector check:")
        print(df.shape[0])
        # print(df.head(5))
        print(res)

    res = con.sql(f"SELECT count(*) FROM {table_name}")
    logger.debug(f" Total records: {res.fetchone()[0]}")


def db_analysis(con, industries="", tickers="", detail=""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    table_name = "tickers_data"  # 'tickers_info' 'tickers_prices'

    if industries:
        filters = f"WHERE sector in ({industries}) "
    else:
        filters = ""

    if tickers:
        if filters == "":
            filters = "WHERE "
        else:
            filters += " AND "
        # filters += f" {table_name}.ticker in ({tickers}) "
        filters += f" ticker in ({tickers}) "

    print(f"\ndb_analysis {table_name} {filters}:")

    if industries == "'Market Indexes'" or detail == "ticker":
        query = f"SELECT date_part('year', date) as year, ticker, AVG(close) as avg_price \
                    FROM {table_name} {filters} \
                    group by ticker, year;"
        res = con.sql(query)
        hue = "ticker"
    else:
        query = f"SELECT date_part('year', date) as year, sector, AVG(close) as avg_price \
                    FROM {table_name} {filters} \
                    group by sector, year;"
        res = con.sql(query)
        hue = "sector"

    if detail == "sector":
        hue = "sector"
    if detail == "ticker":
        hue = "ticker"

    df = res.df()
    print(df.shape[0])
    # print(df['sector'].value_counts())
    print(df.head(10))
    # print(res)

    if not os.path.exists(VISUALS_DIR):
        os.makedirs(VISUALS_DIR)

    # palette = sns.color_palette("RdBu", 10)

    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.countplot(x='sector', data=df, ) # palette=palette)
    # title = 'Distribution by sector'

    sns.lineplot(x="year", y="avg_price", hue=hue, data=df, ax=ax)  # , palette=palette
    title = "AVG price Distribution by sector"
    ax.set_title(title)

    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig(VISUALS_DIR + f"{title}.png")


if __name__ == "__main__":
    if not (DUCKDB_CONNECTION):
        logger.error(f"!! Check DUCKDB_CONNECTION settings. Connection failed.\n")
        exit(1)

    table_name = "tickers_prices"  # 'tickers_info' 'tickers_prices'
    if DEBUG:
        con, tables = duckdb_connect(connection=DUCKDB_CONNECTION)
        print(tables)

        # res = con.sql("""
        #     FROM duckdb_views()
        #     SELECT database_name, schema_name, view_name, internal, temporary, column_count
        #     WHERE NOT internal;""")
        #     # SELECT database_name, schema_name, view_name, internal, temporary, column_count  EXCLUDE(sql)
        # print(res)
        # exit()

        # db_tables(con, table_names="'tickers_info', 'tickers_prices', 'tickers_data'")
        # db_columns(con, table_name='tickers_info', with_sample=True)
        # db_columns(con, table_name='tickers_prices', with_sample=True)
        
        # duckdb_transform_data(con, 'tickers_data')
        db_columns(con, table_name='tickers_data', with_sample=True) # transformed
        exit()

        # df = db_get_ticker_records(con, table_name="tickers_prices", ticker="AAPL", start_date=START_DATE)
        df = db_get_ticker_records(con, table_name="tickers_data", ticker="AAPL", start_date="2025-03-21")
        print("db_get_ticker_records:", df)
        # db_analysis(con,
        #         "'Information Technology', 'Health Care', 'Financials', 'Industrials', 'Consumer Discretionary'",
        #         detail="ticker")
        db_analysis(
            con,
            "'Information Technology'",
            "'AAPL', 'MSFT', 'ADBE', 'ORCL', 'AMD', 'CRM'",
            detail="ticker",
        )
        # db_analysis(con, "'Real Estate', 'Utilities', 'Materials', 'Consumer Staples', 'Energy', 'Communication Services'")
        # db_analysis(con, "")
        # db_analysis(con, "'Market Indexes'", "'^FTEU1', '^SPX', '^SP400', '^SP600', '^DJUS'")
        # db_analysis(con, "", "'^SPX', '^SP400', '^SP600', '^DJUS'")
        # contast:
        # db_analysis(con, "'Consumer Discretionary', 'Utilities', 'Materials', 'Consumer Staples', 'Energy'")
        exit(0)

    path = DATA_DIR
    mask = "tickers_data*.csv"  # .parquet
    try:
        file_list = [sorted(glob(f"{path}{mask}"))[0]]  # only 1st for testing
    except:
        print(f"No {path}{mask} files found.")
        exit(1)
    description = f"Testing loading {mask}"
    # Load files to duckdb
    # duckdb_load_data(df, table_name, description, connection='')
