# Game format V3

## History

* V1: rust writes floats to .bin, python converts to .np
* V2: rust writes floats to .bin.gz, python converts to .hdf5
* V3: design in progress in this spec

## Objectives

* store selfplay and other (cclr) games on disk

* compact (compressed?)

* fast random access

* support different games

* contain as much metadata as possible

* maybe: _gracefully handle input and output encoding changing over time

* immediately write HDF5 files / some other file type from Rust without intermediate recompression in Python

* clearer "game" structure?
    * this is not really what LC0 does, they have a very flat structure instead
    * the advantage is that things like average game length become easy to compute

## Detailed data stored

* metadata json file:

    * game `string`
    * input encoding `string`
    * policy encoding `string`
    * number of games `int`
    * number of positions
    * start offset of each position

* main .bin file:

    * game id

    * moves since start of game`1 float`

    * game length `1 float`c

    * root node visits `1 float`&s

    * available moves

        

    * KDL divergence of wdl (zero || net) `1 float`

    * KDL divergence policy (zero || net) `1 float`

        

    * wdl final `3 floats`

    * wdl zero `3 floats`

    * wdl net `3 floats`

        

    * board state:

        * spatial binary inputs: `N_b x W x W (bits)`
            * pieces, es passant, ...
        * flat number inputs: `N_f (f32)`
            * 50move counter, number of repetitions
            * castling rights, side to move, ...

    * policy zero

        * available move count: `i32`
        * encoded as parse array: `[(i32, f32)]`
        * everything not in the array is an unavailable move, is everything unavailable it's a forced pass move

## Estimated Data Size

* size estimates:
    * chess:
        * per position:
            * position: 20 * 8 * 8 bits = 160 bytes
            * policy: 100 * 2 * 4 bytes = 800 bytes
            * => ~1KB per position
            * => ~1GB per 1M positions

## Infrastructure

* load positions in separate thread while training
* stop using hdf5 and switch back to standard data files

