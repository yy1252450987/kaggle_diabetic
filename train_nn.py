"""Conv Nets training script."""
import click
import numpy as np

import data
import util
from nn import create_net


@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(cnf, weights_from):
    # 配置参数
    config = util.load_module(cnf).config
    # 网络参数路径
    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)
    # 导入训练文件名,以及他们对应的LABEL 
    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names).astype(np.float32)
    # 搭建好网络
    net = create_net(config)
    # 导入训练好的参数到搭建好的网络当中
    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
    # 训练网络
    net.fit(files, labels)

if __name__ == '__main__':
    main()

