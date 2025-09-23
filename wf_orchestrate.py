import warnings

# warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import os
import argparse

from prefect import flow, task # TEMP uncomment

from settings import DATA_DIR, START_DATE, SELECTED_TICKERS, DEBUG, logger
from settings import MODE, SAMPLE, DUCKDB, CHUNKSIZE, SAMPLE_SIZE

# from settings import DEBUG  # isort:skip
# DEBUG = True  # True # False # override global settings

from wf_extract import extract_data
from wf_load_transform import load_data, transform_data
from wf_modeling import train_models
from wf_simulation import simulate

###########################


@task(retries=3)
def extract(mode=MODE, selected_tickers=SELECTED_TICKERS):
    logger.info(f"\nExtracting data: {mode}")
    df_tickers_info, df_tickers = extract_data(mode, selected_tickers)
    return df_tickers_info, df_tickers


@task(retries=3)
def load(df_tickers_info, df_tickers, mode=MODE):
    logger.info(f"\nLoading data: {mode}")
    res = load_data(df_tickers_info, df_tickers, mode, selected_tickers=SELECTED_TICKERS)
    return res


@task(retries=3)
def transform(mode=MODE):
    logger.info(f"\nTransforming data: {mode}")

    if mode==DUCKDB:
        res = transform_data(mode)
    else:
        return 0
    
    return res


@task(retries=3)
def train(mode=MODE):
    logger.info(f"\nTrain models: {mode}")

    if mode==DUCKDB:
        res = train_models(mode)
    else:
        return 0
    
    return res


@task(retries=3)
def predict(mode=MODE):
    logger.info(f"\nSimulate: {mode}")

    if mode==DUCKDB:
        res = simulate(mode)
    else:
        return 0
    
    return res


###########################

## Data Engineering & ML workflow

@flow(log_prints=True)
def ml_workflow(params):

    logger.info(f"\nStarting workflow...")
    mode = params.mode

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Stage: EXTRACT
    # input:
    #  - selected_tickers,
    #  - (?date=current)
    #    if DW already exist we can get last loaded date and backfill only since then
    #  - mode
    df_tickers_info, df_tickers = extract(mode, selected_tickers=SELECTED_TICKERS)
    if mode == SAMPLE:
        # done - no load/transform
        logger.info(f"\nWorkflow finished.")
        return 0

    # Stage: LOAD
    # input:
    #  - mode
    #  TODO - backfill or full load
    columns = ["date","ticker","open","high","low","close","volume"]
    res = load(df_tickers_info, df_tickers[columns], mode)

    # Stage: TRANSFORM
    # input:
    #  - mode
    if res == 0:
        transform(mode)
    else:
        raise ValueError("!! LOADING/TRANSFORMATION FAILED! Stopping.")
    
    # Stage: MODEL TRAINING
    # input:
    #  - mode
    if res == 0:
        train(mode)
    else:
        raise ValueError("!! MODEL TRAINING FAILED! Stopping.")
    
    # Stage: PREDICTION
    # input:
    #  - mode
    if res == 0:
        predict(mode)
    else:
        raise ValueError("!! PREDICTION FAILED! Stopping.")
    
    logger.info(f"\nWorkflow finished.")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process (ELT) data into DuckDb or export sample CSV")

    parser.add_argument("--mode", required=False, type=str, default=MODE, help=f"{SAMPLE} or {DUCKDB} as default")
    # parser.add_argument('--reset', required=False, type=str, default='False', help='True to reset table before loading, False as default')

    args = parser.parse_args()

    # execute Workflow
    res = ml_workflow(args)

    exit(res)
