#!/bin/bash

set -x

MODEL=$1
VISITS=$2
echo Using model: $MODEL
parallel --no-notice -j20 --ungroup ./generate_training.py --use-rpc --die-if-present games/$MODEL/die --network $MODEL --visit-count $VISITS --group-index ::: {1..20}

