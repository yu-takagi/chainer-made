"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import Chain, Variable
from chainer.serializers import save_hdf5, load_hdf5

from made import MADE
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
def run_epoch(split, upto=None, gpu=None):
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    xp = cuda.get_array_module(x)
    N,D = x.shape
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):
        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = xp.zeros(xb.data.shape,dtype=xp.float32)
        if gpu is not None:
            xbhat = cuda.to_gpu(xbhat)
        xbhat = Variable(xbhat)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model.forward(xb)

        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = binary_cross_entropy_with_logits(xbhat, xb) / B
        lossf = loss.data
        if gpu is not None:
            lossfs.append(cuda.to_cpu(lossf))
        else:
            lossfs.append(lossf)

        # backward/update
        if split == 'train':
            model.cleargrads()
            loss.backward()
            opt.update()

    print("%s epoch average loss: %f" % (split, np.mean(lossfs)))

# ------------------------------------------------------------------------------
def run_gen(nb_samples=5,seed=0,gpu=None):
    print("generate %d samples" % nb_samples)
    samples = model.gen(nb_samples=nb_samples,seed=seed,gpu=gpu)
    return samples

# ------------------------------------------------------------------------------
def binary_cross_entropy_with_logits(x, t):
    max_val = F.clip(-x,x_min=0.,x_max=np.inf)
    loss = x - x * t + max_val + F.log(F.exp(-max_val) + F.exp(-x - max_val))
    return F.sum(loss)

# ------------------------------------------------------------------------------
def plot(x,name):
    print("plotting...")
    width = x.shape[0]
    height = 1
    fig, axis = plt.subplots(height, width, sharex=True, sharey=True)

    for i, image in enumerate(x):
        ax = axis[i]
        ax.imshow(image, cmap=plt.cm.gray)
        ax.axis('off')

    plt.savefig(name+'.png')

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to binarized_mnist.npz")
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    parser.add_argument('-g', '--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--gen', type=int, default=None, help="How many samples will be generated")

    args = parser.parse_args()
    # --------------------------------------------------------------------------
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    # reproducibility is good
    np.random.seed(42)

    # load the dataset
    print("loading binarized mnist from", args.data_path)
    mnist = np.load(args.data_path)
    xtr, xte = mnist['train_data'], mnist['valid_data']

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.shape[1], hidden_list, xtr.shape[1], num_masks=args.num_masks, gpu=args.gpu)
    if args.gpu is not None:
        model.to_gpu(args.gpu)
        xtr = cuda.to_gpu(xtr)
        xte = cuda.to_gpu(xte)

    # set up the optimizer
    opt = chainer.optimizers.Adam(alpha=1e-3,weight_decay_rate=1e-4)
    opt.setup(model)

    # start the training
    for epoch in range(100):
        print("epoch %d" % (epoch, ))
        run_epoch('test', gpu=args.gpu, upto=5) # run only a few batches for approximate test accuracy
        run_epoch('train', gpu=args.gpu)
        if epoch % 5 == 0 and args.gen is not None :
            samples = run_gen(nb_samples=args.gen,seed=epoch,gpu=args.gpu)
            samples_rs = np.reshape(samples, (args.gen, 28, 28))
            plot(samples_rs,str(epoch))

    print("optimization done. full test set eval:")
    run_epoch('test', gpu=args.gpu)

