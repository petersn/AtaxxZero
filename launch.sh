#!/bin/bash

set -x

MODEL=$1
echo Using model: $MODEL
parallel --no-notice -j20 --ungroup ./generate_training.py --network $MODEL --group-index ::: {1..20}

