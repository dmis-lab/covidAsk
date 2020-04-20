#!/bin/bash
# Run this file after phrase dumping is done.
DUMP_DIR="dumps_new/denspi_2020-04-10"

# Build document tfidf
python build_doc_tfidf.py "data/2020-04-10" "$DUMP_DIR"

# Build paragraph tfidf
python build_par_tfidf.py "$DUMP_DIR/phrase" "$DUMP_DIR/tfidf" --ranker_path "$DUMP_DIR/2020-04-10-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz" --start "0" --end "4"

# Create index
python run_index.py "$DUMP_DIR" all --hnsw --num_clusters "16384" --cuda
