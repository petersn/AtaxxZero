#!/bin/bash

python3 client.py -e "./BEST.sh" -U "AtaxxZero-f=128-b=12-m=177-v100" -P $(cat ataxx_server_password)
#python3 client.py -e "./example_engine.py" -U "GreedyAtaxx-plies=2" -P $(cat ataxx_server_password)

