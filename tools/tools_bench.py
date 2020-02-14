import argparse
import os
import random

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmaction import __version__
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMAction-Lite Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    dataset = datasets

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    data_loader = data_loaders[0]

    prog_bar = mmcv.ProgressBar(len(dataset[0]))
    for i, data in enumerate(data_loader):
        for img in data['imgs']:
            mean = torch.FloatTensor(cfg.img_norm_cfg['mean']).cuda()
            std = torch.FloatTensor(cfg.img_norm_cfg['std']).cuda()

            imgs = img.cuda()
            imgs = imgs.float()
            # imgs.float()
            imgs.sub_(mean[None, :, None, None, None]).div_(std[None, :, None,
                                                                None, None])
            prog_bar.update()


if __name__ == '__main__':
    main()
