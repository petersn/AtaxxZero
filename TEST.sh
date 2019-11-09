#!/bin/bash

time python3 uai_ringmaster.py \
	--pgn-out evals/turbo_tc5_model-173.pgn \
	--tc 5 \
	--max-plies 400 \
	--show-games \
	--engine "tiktaxx" \
	--engine "python3 ./uai_interface.py --network-path /tmp/model-173.npy"

