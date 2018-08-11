# MADE implemented with Chainer
Implemenation of [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) with [Chainer](https://chainer.org/). This repository's aim is to adapt [A. Karpathy's MADE codes](https://github.com/karpathy/pytorch-made) from PyTorch to chainer. I newly implemented sampling function.

This is the [blog post (only in Japanese)](http://tk-g.hatenablog.jp/) about this repository.

You can also find the author's original code [here](https://github.com/mgermain/MADE).

MIT license. Contributions welcome.

## Requirements
python 2.x, chainer 4.3.1, numpy, matplotlib, and [binarized mnist dataset](https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz).

## examples
Training a 1-layer MLP of 500 units with only a single mask, and using a single fixed (but random) ordering as so:

```
python run.py --data-path binarized_mnist.npz -q 500
```

which converges at binary cross entropy loss of `94.06`.

We can use 10 orderings (`-n 10`) and also average over the 10 at inference time (`-s 10`):

```
python run.py --data-path binarized_mnist.npz -q 500 -n 10 -s 10
```

which gives a much better test loss of `83.08`.

## TODO
* GPU
* Condtion