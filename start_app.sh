#!/bin/bash
echo
echo '1. STARTING WORKFLOW'
echo

set DATAWAREHOUSE=DUCKDB

echo
echo '1.1. WORKFLOW ORCHESTRATE'
python wf_orchestrate.py --mode=$DATAWAREHOUSE

# echo
# echo '2. Starting Streamlit app...'
# echo
# streamlit run dashboard-app.py

sleep 5
