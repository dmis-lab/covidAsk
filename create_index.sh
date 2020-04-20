#!/bin/bash
# Run this file after phrase dumping is done.

# Build document tfidf
python build_doc_tfidf.py data/2020-04-10 dumps_new/denspi_2020-04-10

# Build paragraph tfidf
python build_par_tfidf.py dumps_new/denspi_2020-04-10/phrase dumps_new/denspi_2020-04-10/tfidf

# Create index
python run_index.py dumps_new/denspi_2020-04-10 all --hnsw --num_clusters "16384" --cuda
