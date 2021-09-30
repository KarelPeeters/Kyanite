# Ideas

This file contains a bunch of ideas that might be interesting to try in the future.

## Search

* first play urgency

* change cpuct to trust value more when a node has been visited a lot

* treat repeated positions as draws (this is not really part of a game necessarily, but something extra in zero search)

## Selfplay

* find balanced opening book automatically
* KLD thresholding, visit nodes where zero and net policy diverge a lot more
* resign games if zero things the winning chance is super low for N consecutive moves, with a threshold automatically determined so errors are rare
* think about ways to not have temperature to cause lower quality game values
* think about alternatives to Dirichlet, the whole justification for it is shaky at best
* output games in the order they were started in to ensure there is no bias towards short game every time the selfplay server is restarted

## Network architecture

* squeeze-excitation layers
* fixup initialization
* more compact policy representations
    * see LC0 discord for a couple suggestions
* bottleneck blocks
* look at mobilenet V1 V2 (V3)
* Ghost/Virtual Batch norm: use smaller batches to get batch statistics, supposed to solve the very real "train/eval" gap
* figure out why the value head is so bad at fitting the data

## Network evaluation performance

* weight quantization

* check whether we're actually using the full capacity of the GPU right now, try with smaller IO data to make sure

## Training 

* stochastic weight averaging

* learning rate schedule

* cyclic learning rate

* use value estimate as training target instead of (only) the final value

## Loop infrastructure

* settle on a common data format, probably hdf5 straight from rust
    * boards and policy should be represented compactly, right now we're wasting massive amounts of space and compensating for it with compression

* estimate Elo during loop, and only switch network if it actually turns out to be better?
    * this involves writing a bot vs bot tool that can actually batch GPU requests

* increase network size over time

* allow for changing loop params at runtime (in the GUI?)

* allow selfplay and training to run on different computers over the real internet

## Infrastructure

* write a GUI to look at boards and see the zero search happening in real time

## Dependencies

* look into removing rust dependencies, the alphazero crate is getting heavy

* properly implement cross-platform cudnn header finding (with env var)


## Other projects to "borrow" ideas from:

* AlphaZero.jl
* KataGo
* LeelaZero
* LeelaChessZero