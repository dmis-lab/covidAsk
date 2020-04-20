#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

DUMP_DIR="dumps/denspi_2020-04-10"
Q_PORT="9010"
D_PORT="9020"
PORT="9030"

# Serve query encoder / metadata / phrase dump
nohup python covidask.py --run_mode "q_serve" --parallel --cuda --query_encoder_path "models/denspi/1/model.pt" --query_port "$Q_PORT" > "$LOG_DIR/q_serve_$Q_PORT.log" &
nohup python covidask.py --run_mode "d_serve" --dump_dir "$DUMP_DIR" --doc_ranker_name "COVID-abs-v1-ner-norm-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz" --doc_port "$D_PORT" > "$LOG_DIR/d_serve_$D_PORT.log" &
nohup python covidask.py --run_mode "p_serve" --aggregate --dump_dir "$DUMP_DIR" --query_port "$Q_PORT" --doc_port "$D_PORT" --index_port "$PORT" > "$LOG_DIR/p_serve_$PORT.log" &

echo "Serving covidAsk. Will take a minute."
