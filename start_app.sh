#!/bin/bash
echo
echo '1. STARTING WORKFLOW'
echo

export DATAWAREHOUSE=duckdb
export DUCKDB_CONNECTION=data/stocks.db
echo $DATAWAREHOUSE
echo $DUCKDB_CONNECTION

echo
echo '1.1. WORKFLOW ORCHESTRATE'
python wf_orchestrate.py --mode=$DATAWAREHOUSE

# echo
# echo '2. Starting Streamlit app...'
# echo
# streamlit run dashboard-app.py

sleep 5
