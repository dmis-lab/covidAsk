#!/bin/bash
# Run this file after phrase dumping is done.
DUMP_DIR="dumps_new/denspi_2020-04-10"

# Build document tfidf
python build_doc_tfidf.py "data/2020-04-10" "$DUMP_DIR"

# Build paragraph tfidf
python build_par_tfidf.py "$DUMP_DIR/phrase" "$DUMP_DIR/tfidf"

# Create index
python run_index.py "$DUMP_DIR" all --hnsw --num_clusters "16384" --cuda
