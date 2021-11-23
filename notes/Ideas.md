# Ideas

This file contains a bunch of ideas that might be interesting to try in the future.

## Search

* first play urgency
* change cpuct to trust value more when a node has been visited a lot
* treat repeated positions as draws (this is not really part of a game necessarily, but something extra in zero search)
* look into the issue of policy being encouraged to diverge in selfplay, and figure out a fix

## Selfplay

* find balanced opening book automatically
* KLD thresholding, visit nodes where zero and net policy diverge a lot more
* resign games if zero things the winning chance is super low for N consecutive moves, with a threshold automatically determined so errors are rare
* think about ways to not have temperature to cause lower quality game values
* think about alternatives to Dirichlet, the whole justification for it is shaky at best
* output games in the order they were started in to ensure there is no bias towards short game every time the selfplay server is restarted

## Network architecture

### Input

* add available moves as output for regularization
* add available moves as input for value head improvement

### Tower

* squeeze-excitation layers
* fixup initialization
* bottleneck blocks
* look at mobilenet V1 V2 (V3)
* Ghost/Virtual Batch norm: use smaller batches to get batch statistics, supposed to solve the very real "train/eval" gap
* attention blocks
* add more padding, maybe so the intermediate layers are 10x10 or even 16x16 (with initial dilations)
* autoencoder style compression and decompression with same-shape skip connections

### Value head

* figure out why the value head is so bad at fitting the data
  * it's not the BN within the value head
  * it still feels like a BN issue because switching to eval mode during training has a large effect
  * maybe BN is being abused to pass global info?
* experiment more with global pooling vs flattening

### Policy head

* more compact policy representations
    * see LC0 discord for a couple suggestions
* attention policy head
    * with and without separated queen moves
* try flat policy head

## Network inference performance

* weight quantization

* check whether we're actually using the full capacity of the GPU right now, try with smaller IO data to make sure

## Training 

* stochastic weight averaging
* learning rate schedule
* cyclic learning rate
* use value estimate as training target instead of (only) the final value, see https://ala2020.vub.ac.be/papers/ALA2020_paper_18.pdf
* policy learning
* label smoothing (don't push the network to inf in softmax output layers)
* supervised: use stockfish evals in pgn games instead of the final value

## Loop infrastructure

* estimate Elo during loop, and only switch network if it actually turns out to be better?
    * this involves writing a bot vs bot tool that can actually batch GPU requests
* increase network size over time
* allow selfplay and training to run on different computers over the real internet
* allow interactive parameter updates without having to restart, maybe with a config file
* make it easier to keep multiple config combinations around without hardcoding everything in python

## Infrastructure

* write a GUI to look at boards and see the zero search happening in real time

## Dependencies

* look into removing rust dependencies, the alphazero crate is getting heavy
  * it's getting worse and worse ...
* properly implement cross-platform cudnn header finding (with env var)


## Other projects to "borrow" ideas from:

* AlphaZero.jl
* KataGo
* LeelaZero
* LeelaChessZero