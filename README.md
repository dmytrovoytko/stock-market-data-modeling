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

What ML models would be able to predict market trends? which model would work better?

Many questions...

So I decided to better understand it by "playing" with data the way I want, not as investors channels present us. And let's my MLOps skills help me with that!

I decided to collect historical data of stocks, enrich it, put it into convenient portable warehouse (DuckDB) and experiment with modeling using different ML classifiers (RandomForest, DecisionTree, AdaBoost). Then compare results.
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

## 🚀 Instructions to reproduce

- [Setup environment](#hammer_and_wrench-setup-environment)
- [Run workflow](#arrow_forward-run-workflow)
- [Modeling results](#mag_right-modeling-results)

### :hammer_and_wrench: Setup environment

1. **Fork this repo on GitHub**. Or use `git clone https://github.com/dmytrovoytko/stock-market-data-modeling.git` command to clone it locally, then `cd stock-market-data-modeling`.
2. Create GitHub CodeSpace from the repo.
3. **Start CodeSpace**
4. The app works in docker container, **you don't need to install packages locally to test it**.
5. Only if you want to develop the project locally, you can run `pip install -r requirements.txt` (project tested on python 3.12).

By default workflow will use simple portable warehouse - DuckDb database (you can also create/use free! [MotherDuck](https://motherduck.com/) account to use cloud data warehouse).


### :arrow_forward: Run workflow

#### Run the project locally / in Codespaces

1. Install packages
`pip install --no-cache-dir -r requirements.txt`

2. Run the workflow
`bash start_app.sh`
which sets environment and executes `wf_orchestrate.py`

#### Run the project in Docker (as well works in Codespaces)

1. **Run `bash run-docker.sh` to nuild and start app container**. As packages include some quite heavy packages like Prefect, building container takes some time (~3min). 

![docker-build](/screenshots/docker-00.png)

When you see these messages the workflow has started

![docker-run](/screenshots/docker-01.png)

Tickers data is being downloaded

![docker-run](/screenshots/docker-02.png)

If you want to experiment with different stocks, edit `ticker_sets` in `settings.py`, and then run the workflow again. 

### :mag_right: Modeling results

After running the workflow resulted models can be found in `data` folder:
- `RandomForest_model.joblib` and `RandomForest.json`
- `DecisionTree_model.joblib` and `DecisionTree.json`
- `adaboost_model.joblib` and `adaboost.json`

![docker-run](/screenshots/modeling-results.png)

and different financial metrics in .json format, for example (in files data is formatted for readability)

![docker-run](/screenshots/fin-metrics-0.png)

![docker-run](/screenshots/fin-metrics-1.png)

Also in logs you can find correlations of features for each classifier, for example

![docker-run](/screenshots/features-correlations.png)

Comparing financial metrics we can decide which model words better for selected set of tickers.

Simulation gives opportunity to see results of the 30-day prediction estimation like this

![docker-run](/screenshots/modeling-prediction.png)

Information about stocks can be found in 2 files:
- `snp500.csv` with the list of companies, sectors and tickers
- `tickers_prices.csv` with tickers prices + some calculated features like ma_10,ma_20,ema_20,momentum,volatility,next_day_return

This gives opportunity to check what stocks had lower/higher volatility during last months.

All this information makes you feel more comforatble in making decision what companies consider for investing.


## 🔜 Next steps

Stock/Crypto Analitics is a very interesting topic, especially now when market grows after months of panic and high volatility!

I plan to analyze more tecnical indicators, visualize data and use more advanced models to play with predictions, I think it would be interesting!

Stay tuned!


## Support

🙏 Thank you for your attention and time!

- If you experience any issue while following this instruction (or something left unclear), please add it to [Issues](/issues), I'll be glad to help/fix. And your feedback, questions & suggestions are welcome as well!
- Feel free to fork and submit pull requests.

If you find this project helpful, please ⭐️star⭐️ my repo 
https://github.com/dmytrovoytko/stock-market-data-engineering to help other people discover it 🙏

Made with ❤️ in Ukraine 🇺🇦 Dmytro Voytko