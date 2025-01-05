#!/bin/bash

echo "Listing /app contents:"
ls -l /app

while [ ! -f "/app/datos/processed_alimentacion.csv" ]; do
  sleep 1
done

while [ ! -f "/app/datos/processed_salud.csv" ]; do
  sleep 1
done

while [ ! -f "/app/datos/processed_vivienda.csv" ]; do
  sleep 1
done

while [ ! -f "/app/datos/processed_vestimenta.csv" ]; do
  sleep 1
done

while [ ! -f "/app/datos/processed_EAC.csv" ]; do
  sleep 1
done

while [ ! -f "/app/datos/processed_test_favorita.csv" ]; do
  sleep 1
done

echo "All files are ready. Starting the main script..."

# you can remove the files after reading them
# rm -f "/app/data/embeddings.csv"
# rm -f "/app/data/embeddings_test.csv"

# Run the actual command (e.g., start cluster.py)
exec "$@"
