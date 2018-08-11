"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Yu Takagi based on https://arxiv.org/abs/1502.03509 and https://github.com/karpathy/pytorch-made
"""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable
from chainer.functions.connection import linear
from chainer import cuda

# ------------------------------------------------------------------------------

class MaskedLinear(L.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_size=in_features, out_size=out_features)

    def set_mask(self, mask, gpu):
        mask = mask.T.astype(np.uint8)
        if gpu is not None:
            mask = cuda.to_gpu(mask)
        self.mask = mask

    def __call__(self, x):
        return linear.linear(x, self.W*self.mask, self.b)

class MADE(chainer.Chain):

    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, gpu=None):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        gpu: GPU ID (None indicates CPU)
        """
        super(MADE, self).__init__()
        with self.init_scope():
            self.nin = nin
            self.nout = nout
            self.hidden_sizes = hidden_sizes
            assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
            self.gpu = gpu

            # define a simple MLP neural net
            net = []
            hs = [nin] + hidden_sizes + [nout]
            for h0,h1 in zip(hs, hs[1:]):
                net.extend([
                        MaskedLinear(h0, h1),
                        F.relu,
                    ])
            net.pop() # pop the last ReLU for the output layer
            self.net = chainer.Sequential(*net)

            # seeds for orders/connectivities of the model ensemble
            self.natural_ordering = natural_ordering
            self.num_masks = num_masks
            self.seed = 0 # for cycling through num_masks orderings
            self.m = {}
            self.update_masks() # builds the initial self.m connectivity
            # note, we could also precompute the masks and cache them, but this
            # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        self.input_order = self.m[-1]
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m,self.gpu)

    def forward(self, x):
        return self.net(x)

    def gen(self, nb_samples=1, seed=0, gpu=None):
        swap_order = self.input_order
        input_size = self.net[0].W.shape[1]
        samples = np.zeros((nb_samples, input_size),dtype=np.float32)
        if gpu is not None:
            samples = cuda.to_gpu(samples)
        xp = cuda.get_array_module(samples)
        rng = np.random.RandomState(self.seed+seed)

        for i in range(input_size):
            inv_swap = np.where(swap_order == i)[0][0]
            out = self.forward(samples)
            out_exp = F.exp(out[:, inv_swap])
            out_exp_invsig = F.log(out_exp/(1-out_exp))
            out_exp_invsig = F.clip(out_exp_invsig,x_min=0.,x_max=1.)
            out_exp_invsig.data[xp.isnan(out_exp_invsig.data)] = 1

            if gpu is not None:
                prob = cuda.to_cpu(out_exp_invsig.data)
            else:
                prob = out_exp_invsig.data
            sample = rng.binomial(p=prob, n=1)

            if gpu is not None:
                sample =cuda.to_gpu(sample)
            samples[:, inv_swap] = sample

        if gpu is not None:
            samples = cuda.to_cpu(samples)
        return samples

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # run a quick and dirty test for the autoregressive property
    D = 10
    rng = np.random.RandomState(14)
    x = (rng.rand(1, D) > 0.5).astype(np.float32)

    configs = [
        (D, [], D, False),                 # test various hidden sizes
        (D, [200], D, False),
        (D, [200, 220], D, False),
        (D, [200, 220, 230], D, False),
        (D, [200, 220], D, True),          # natural ordering test
        (D, [200, 220], 2*D, True),       # test nout > nin
        (D, [200, 220], 3*D, False),       # test nout > nin
    ]

    for nin, hiddens, nout, natural_ordering in configs:

        print("checking nin %d, hiddens %s, nout %d, natural %s" %
             (nin, hiddens, nout, natural_ordering))
        model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering)

        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nout):
            xtr = Variable(x)
            xtrhat = model.forward(xtr)
            loss = xtrhat[0,k]
            loss.backward()

            depends = (xtr.grad[0] != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))

