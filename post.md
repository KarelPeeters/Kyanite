# Announcing Kyanite - A library for neural network inference

https://github.com/KarelPeeters/Kyanite/

<hr>

# How to fix docs.rs for sys crate that depends on system libraries? 

[//]: # (current idea: vendor headers and use those for docs)

How do I go about generating working docs for a -sys crate that wraps a bunch of cuda libraries? This is my failed build log: https://docs.rs/crate/kn-cuda-sys/0.2.1/builds/921381
I don't think I'm allowed to "vendor" the headers and libraries into my own github repo by nvidia
I read https://docs.rs/about/builds#missing-dependencies, but I'm not sure if adding the cuda libraries to the build env makes sense? They change version often, are not all that backwards compatible, and the installation process is a convoluted mess of manually unzipping files or dealing with dpkg, eg.https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux
Can I build the docs locally and publish that somehow? Or is there a solution I'm not seeing here?