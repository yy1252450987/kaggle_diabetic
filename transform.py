from __future__ import division
import importlib
import time

import click
import numpy as np

import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu1") # doesn't seem to work when set here
#import theano

import nn, util, iterator

from definitions import *
import tta

@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
@click.option('--n_iter', default=1,
              help="Iterations for test time averaging.")
@click.option('--skip', default=0,
              help="Number of pseudo TTA iterations to skip.")
@click.option('--test', is_flag=True, default=False)
@click.option('--train', is_flag=True, default=False)
@click.option('--weights_from', default=None,
              help='Path to initial weights file.', type=str)
def transform(cnf, n_iter, skip, test, train, weights_from):

    model = util.load_module(cnf).model

    runs = {}
    if train:
        runs['train'] = model.get('train_dir', TRAIN_DIR)
    if test:
        runs['test'] = model.get('test_dir', TEST_DIR)

    model.cnf['batch_size_test'] = 128

    # reduced augmentation for TTA
    #model.cnf['sigma'] = 0.02
    #model.cnf['color'] = False
    #model.cnf['aug_params']['zoom_range'] = (1.0 / 1.2, 1.2)
    #model.cnf['aug_params']['translation_range'] = (20, 20)
    #start = model.cnf['slope'].get_value()
    #model.cnf['slope'].set_value(nn.float32(
    #    start * model.cnf['slope_decay'] ** 209))
    #print(model.cnf['slope'].get_value())

    net = nn.create_net(model, tta=True if n_iter > 1 else False)

    if weights_from is None:
        net.load_params_from(model.weights_file)
        print("loaded weights from {}".format(model.weights_file))
    else:
        weights_from = str(weights_from)
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))

    tfs, color_vecs = tta.build_quasirandom_transforms(
            n_iter, model.cnf['sigma'], skip=skip, **model.cnf['aug_params'])

    for run, directory in sorted(runs.items(), reverse=True):

        print("transforming {}".format(directory))
        tic = time.time()

        files = util.get_image_files(directory)

        Xs, Xs2 = None, None

        for i, (tf, color_vec) in enumerate(zip(tfs, color_vecs), start=1):

            print("{} transform iter {}".format(run, i))

            X = net.transform(files, transform=tf, color_vec=color_vec)
            if Xs is None:
                Xs = X
                Xs2 = X**2
            else:
                Xs += X
                Xs2 += X**2

            print('took {:6.1f}s'.format(time.time() - tic))
            if i % 10 == 0 or n_iter < 5:
                std = np.sqrt((Xs2 - Xs**2 / i) / (i - 1))
                model.save_transform(Xs / i, i, skip=skip,
                                     test=True if run == 'test' else False)
                model.save_std(std, i, skip=skip,
                               test=True if run == 'test' else False)
                print('saved {} iterations'.format(i))
                Xs, Xs2 = None, None

if __name__ == '__main__':
    transform()
