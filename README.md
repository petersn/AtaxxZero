## AtaxxZero

An AlphaZero style engine for [Ataxx](https://en.wikipedia.org/wiki/Ataxx).

First you must compile the accelerated game generation binary in `cpp/`:

```
    $ cd cpp/
	$ make
```

TODO: Document dependencies to compile.

To get the process started we need to generate an initial model.
Our very first "model" will be uniformly random play.
Start by making game and model directories, and then generate 2000 random-play games.

```
    $ mkdir run1 run1/games run1/models
    $ ./generate_games.py --random-play --output-games run1/games/random-play.json --game-count 2000
```

This will have written a `.json` file to `run1/games/random-play.json` that contains 2000 uniformly random games.
We can now build our first network off of this data.
The policy part of the network is basically useless as it's trained off of uniformly random moves (except that it has some understanding of move legality), but the value component is marginally useful, because even under uniformly random play it's better to have more pieces.

```
    $ ./train.py --games run1/games/random-play.json --steps 2000 --new-path run1/models/model-001.npy
```

Once you have this network you can launch the main process that handles self-improvement:

```
    $ ./looper.py --prefix run1/
```

On my GTX 970 it takes a couple of minutes per self-improvement loop with stock settings.

## License

All of the code is public domain (or CC0, for those outside of the United States), so you may do whatever you want with it, with the exception of the files `cpp/{ataxx,bitboards,invalid,makemove,move,movegen,other}.{cpp,hpp}`, which are a 7x7 Ataxx movegen and bitboard implementation taken from https://github.com/kz04px/ataxx-engine, and are MIT licensed.

