# kZero

An implementation of the AlphaZero paper for different board games, implemented in a combination of Python and Rust.

The specific board games currently implemented are Chess, Ataxx and SuperTicTacToe. Adding a new game should be relatively easy. Currently this repository is very experimental and not very portable and easy-to-use.

The basic file structure is as follows:

* `python` contains the training code, which uses the PyTorch framwork to train networks
    * `loop_main.py` is the entry point of the training loop
* `rust` is a workspace consisting of the following crates:
    * `cuda-sys` contains Rust wrappers for CuDNN, generated at compile time based on system headers
    * `cuda-nn-eval` is a barebones custom CuDNN-based neural network evaluator that takes in onnx files and can pretty much only handle resnet
    * `alpha-zero` is the actual AlphaZero and selfplay server implementation

During a training loop the python training framwork connects to the selfplay TCP server running on localhost and gives it some initial settings. The selfplay server writes generated games to a file, and when it has generated enough games it signals this to the training framework. The network is then trained on these new games, and when training finishes the new network is sent to the selfplay server.

Some effort has been put into optimizing selfplay, so network evaluations are batched and multiple threads and GPUs can be used, although the latter is not frequently tested.
