# Stock Market Analytics/Crypto Machine Learning project 

Machine Learning (& MLOps) project for Stock Markets Analytics Zoomcamp 2025: Stock prices -> DuckDB + ML Modeling & Simulation

ELT workflow for 💹 [Yahoo Finance](https://finance.yahoo.com/markets/)

![Stock Market Analytics/Crypto Machine Learning project](/screenshots/stock-market-data-modeling.png)
Project can be tested and deployed in **GitHub CodeSpaces** (the easiest option, and free), or just locally.
For the GitHub CodeSpace option you don't need to use anything extra at all - just your favorite web browser + GitHub account is totally enough.

## Problem statement

Investing in stocks or crypto became a buzz, many of us want to try earning money with that, BUT how do you choose companies/currencies to invest in?
Which of them are good/bad to buy & sell just right now, and which are good to buy and hold for the long term?

My main interest is in The Magnificent 7 stocks - seven of the world's biggest and most influential tech companies: Apple, Microsoft, Amazon, Alphabet (Google)... Should I invest in particular companies (like Magnificent 7) or investing in S&P 500 index funds would be a better/safer choice for me?

So I decide to better understand it by "playing" with data the way we want, not as investors channels present us. And let's MLOps help me with that!

I decided to collect historical data of stocks, enrich it, put it into convenient portable warehouse (DuckDB) and experiment with modeling using different ML classificators (RandomForest, DecisionTree, AdaBoost).
Let's see how well I can deal with that!

## 🎯 Goals

This is my ML/MLOps project in [Stock Markets Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp)'2025.

**The main goal** is straight-forward: build an end-to-end **Extract - Load - Transform** data pipeline, then train and evaluate several models.  
- choose stocks/tickers to analyze
- process (extract, load, transform) data (from Yahoo Finance to DuckDB)
- deploy orchestration tool to manage pipeline
- evaluate metrics of different models

## :toolbox: Tech stack

- Python 3.12
- **Docker** for containerization
- **Prefect** for workflow orchestration
- **DuckDB** (or MotherDuck) for data warehouse

## Run the project locally

1. Install packages
`pip install --no-cache-dir -r requirements.txt`

2. Run the workflow
`bash start_app.sh`
which sets environment and executes `wf_orchestrate.py`

## Run the project in Docker

`bash run-docker.sh` 

## Modeling results

After running the workflow resulted models and financial metrics can be found in `data` folder:
- `RandomForest_model.joblib` and `RandomForest.json`
- `DecisionTree_model.joblib` and `DecisionTree.json`
- `adaboost_model.joblib` and `adaboost.json`


Made with ❤️ in Ukraine 🇺🇦 Dmytro Voytko