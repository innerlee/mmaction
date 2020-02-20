"""This file is for benchmark dataloading process.
The command line to run this file is:
$ python -m cProfile -o program.prof tools/bench_processing.py
config/bench_processing.py

It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse

import mmcv
from mmcv import Config
from mmcv.image import use_backend

from mmaction import __version__
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--jpeg_backend',
        type=str,
        default='cv2',
        choices=['cv2', 'turbojpeg'],
        help='backend for jpeg decoding')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init logger before other steps
    logger = get_root_logger('INFO')
    logger.info('MMAction-Lite Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set jpeg backend
    if args.jpeg_backend is not None:
        use_backend(args.jpeg_backend)

    dataset = build_dataset(cfg.data.train)

    data_loader = build_dataloader(
        dataset,
        cfg.data.videos_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        for img in data['imgs']:
            prog_bar.update()


if __name__ == '__main__':
    main()
