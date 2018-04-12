#!/bin/bash

set -x

MODEL=$1
echo Using model: $MODEL
parallel --no-notice -j20 --ungroup ./generate_training.py --use-rpc --die-if-present games/$MODEL/die --network $MODEL --group-index ::: {1..20}

