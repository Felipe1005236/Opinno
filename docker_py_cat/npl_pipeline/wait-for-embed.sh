#!/bin/bash

# Wait for embeddings.csv to complete
while [ ! -f "/app/data/embeddings.csv" ]; do
  sleep 1
done

# Wait for embeddings_test.csv to complete
while [ ! -f "/app/data/embeddings_test.csv" ]; do
  sleep 1
done

# Wait for keywords embeddings.csv to complete
while [ ! -f "/app/data/embedded_keywords.csv" ]; do
  sleep 1
done

# Optionally, you can remove the files after reading them
# rm -f "/app/data/embeddings.csv"
# rm -f "/app/data/embeddings_test.csv"

# Run the actual command (e.g., start cluster.py)
exec "$@"
