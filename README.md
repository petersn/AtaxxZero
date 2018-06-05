## AtaxxZero

An AlphaZero style engine for [Ataxx](https://en.wikipedia.org/wiki/Ataxx).

To get the process started we need to generate an initial model.
Our very first "model" will be uniformly random play.
Start by making game and model directories, and then generate 2000 random-play games.

```
    $ mkdir games models
    $ python generate_training.py --network random-play --random-play --game-count 2000
```

This will have written a `.json` file in `games/random-play/` that contains 2000 uniformly random games.
We can now build our first network off of this data.
The policy part of the network is basically useless, as it's trained off of uniformly random moves, but the value component is marginally useful (which gets the ball rolling).

```
    $ python train.py --games games/random-play/ --steps 3000 --new-name model-001
```

Once you have this network you can launch the main process that handles self-improvement:

```
    $ python looper.py
```

On my GTX 970 it takes about an hour per self-improvement loop.

## License

All of the code is public domain (or CC0, for those outside of the United States), so you may do whatever you want with it, with the exception of the files `cpp/{ataxx,bitboards,invalid,makemove,move,movegen,other}.{cpp,hpp}`, which are a 7x7 Ataxx movegen and bitboard implementation taken from https://github.com/kz04px/ataxx-engine, and are MIT licensed.

