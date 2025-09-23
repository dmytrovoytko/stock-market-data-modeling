#!/bin/bash
echo
echo '1. DOCKER Build'
echo

export DATAWAREHOUSE=duckdb
export DUCKDB_CONNECTION=data/stocks.db
echo $DATAWAREHOUSE
echo $DUCKDB_CONNECTION

echo
docker build . --tag 'sma_proj'

echo '2. DOCKER Run: WORKFLOW ORCHESTRATE'
echo
docker run  -v ./data:/app/data sma_proj