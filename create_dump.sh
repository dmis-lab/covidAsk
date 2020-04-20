#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

# Each process takes up ~10GB GPU memory. Assign different GPUs if necessary.
CUDA_VISIBLE_DEVICES=0 nohup python local_dump.py --parallel --filter_threshold "-2" --start "0" --end "10" > "$LOG_DIR/0-10.log" &
CUDA_VISIBLE_DEVICES=0 nohup python local_dump.py --parallel --filter_threshold "-2" --start "10" --end "20" > "$LOG_DIR/10-20.log" &
CUDA_VISIBLE_DEVICES=1 nohup python local_dump.py --parallel --filter_threshold "-2" --start "20" --end "30" > "$LOG_DIR/20-30.log" &
CUDA_VISIBLE_DEVICES=1 nohup python local_dump.py --parallel --filter_threshold "-2" --start "30" --end "37" > "$LOG_DIR/30-37.log" &

echo "Dumping phrases"
