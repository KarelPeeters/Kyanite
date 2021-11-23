# Move indexing redesign

Reworking the current move indexing system to more easily support different move head formats.

## Goals

* make it easier to test new policy head architectures
* ... without trying everything to conv indices

## Thoughts

* find a simple (mostly?) flat move indexing scheme for all 1858 different POV moves 
* should castling, pawn moves and promotions be separate?
  * => separate out promotions and castling, keep pawn moves fused into queen moves
* export this indexing scheme to both python and rust, generate from rust?
* use for attention policy head

## Specific design

* flat index: 0..1858
  * queen (including pawns, rook, bishop) 0..1456
  * knight 1456..1792
  * promotions (including queen), from+to+piece #88
  * castling #2
* conv index: match old indexing scheme, and make sure nets still play with the same strength
  * queen (+castling and =q) 8x7 x 8x8
  * knight 8 x 8x8
  * underpromotion 3x3 x 8x8
* attention index
  * 8x8 from
  * 8x8 to 
  * 1x8x4 to_promotion 
    * also try using queen moves as promotions

