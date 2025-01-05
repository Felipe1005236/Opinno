#!/bin/bash

# Wait for processed_data.csv to complete
while [ ! -f "/app/data/processed_data.csv" ]; do
  sleep 1
done

# Wait for processed_data_test.csv to complete
while [ ! -f "/app/data/processed_data_test.csv" ]; do
  sleep 1
done

# Wait for categories to be processed
while [ ! -f "/app/data/processed_data_cat.csv" ]; do
  sleep 1
done

# Optionally, you can remove the files after reading them
# rm -f "/app/data/processed_data.csv"
# rm -f "/app/data/processed_data_test.csv"

exec "$@"
