# Game format

## Objectives

* store games on disk

* compact (compressed?)

* fast random access

* support different games

* contain as much metadata as possible 

* (gracefully handle input and output encoding changing over time)

* immediately write HDF5 files from Rust without intermediate recompression in Python

* clearer "game" structure?
  * this is not really what LC0 does, they have a very flat structure instead
  * the advantage is that things like average game length become easy to compute

## Design

* stats necessary for each position:
  * board state
    * pieces, castling rights, side to move, rule50 count, repetitions
    * history? -> YES, hopefully this file is fast to write :)
  * wdl final
  * wdl estimate
  * policy (0..1, and -1 for masked)
  * KDL divergence wdl
  * KDL divergence policy
  * moves left
  * moves prev (or maybe this should be in board state)

* tree structure:

```
game = "chess"

games/
    count: int
    lengths: [G] int
    position_start

positions/
    board: P*C*8*8, bits
    
    
    
    
```


# Infrastructure

* load positions in separate thread while training